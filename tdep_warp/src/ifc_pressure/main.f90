#include "precompilerdefinitions"
program ifc_pressure
  use konstanter, only: r8, lo_iou, lo_tol, lo_bohr_to_A, lo_pressure_HartreeBohr_to_GPa
  use gottochblandat, only: tochar, walltime, lo_mean, lo_stddev, lo_does_file_exist
  use mpi_wrappers,  only: lo_mpi_helper, lo_stop_gracefully
  use lo_memtracker, only: lo_mem_helper
  use type_crystalstructure, only: lo_crystalstructure
  use type_mdsim, only: lo_mdsim
  use type_forceconstant_secondorder, only: lo_forceconstant_secondorder
  use type_forceconstant_thirdorder,  only: lo_forceconstant_thirdorder
  use type_forceconstant_fourthorder, only: lo_forceconstant_fourthorder
  use symmetrize, only: symmetrize_stress
  use helper, only: write_file_stress_voigt, write_stress_voigt
  use options, only: lo_opts


  implicit none

  type(lo_opts) :: opts
  type(lo_crystalstructure) :: uc, ss
  type(lo_mdsim) :: sim
  type(lo_forceconstant_secondorder) :: fc2, fc2_ss
  type(lo_forceconstant_thirdorder)  :: fc3, fc3_ss
  type(lo_forceconstant_fourthorder) :: fc4, fc4_ss
  type(lo_mpi_helper) :: mw
  type(lo_mem_helper) :: mem

  logical :: have_fc3, have_fc4
  integer :: t, i, j
  real(r8) :: t0, volume, n_samples

  ! time series tensors (small: 3x3xnt)
  real(r8), allocatable :: s0t(:,:,:), s2t(:,:,:), s3t(:,:,:), s4t(:,:,:), srt(:,:,:)

  ! mean/std/err
  real(r8) :: s0(3,3), s2(3,3), s3(3,3), s4(3,3), sr(3,3)
  real(r8) :: s0_std(3,3), s2_std(3,3), s3_std(3,3), s4_std(3,3), sr_std(3,3)
  real(r8) :: s0_err(3,3), s2_err(3,3), s3_err(3,3), s4_err(3,3), sr_err(3,3)

  ! diagnostic: DFT - FC2 only (is the FC2 subtraction a variance-REDUCING swap?)
  real(r8) :: d02(3,3), d02_std(3,3), d02_err(3,3)
  real(r8) :: d02_sym(3,3), d02_std_sym(3,3), d02_err_sym(3,3)

  ! symmetrized
  real(r8) :: s0_sym(3,3), s2_sym(3,3), s3_sym(3,3), s4_sym(3,3), sr_sym(3,3)
  real(r8) :: s0_std_sym(3,3), s2_std_sym(3,3), s3_std_sym(3,3), s4_std_sym(3,3), sr_std_sym(3,3)
  real(r8) :: s0_err_sym(3,3), s2_err_sym(3,3), s3_err_sym(3,3), s4_err_sym(3,3), sr_err_sym(3,3)

  ! local accumulators per timestep
  real(r8) :: s2loc(3,3), s3loc(3,3), s4loc(3,3)
  real(r8) :: v0(3), u1(3), u2(3), u3(3), u4(3)
  real(r8) :: m2(3,3), m3(3,3,3), m4(3,3,3,3)
  real(r8) :: rv2(3), rv3(3), rv4(3)
  integer :: a1, a2, a3, a4, ii, i1, i2, i3, i4, i5

  ! for compliance tensore
  real(r8) :: Svoigt(6,6)
  integer  :: iu

  t0 = walltime()
  call mw%init()
  call mem%init()
  call opts%parse()
  if (.not. mw%talk) opts%verbosity = -100

  if (mw%talk) write(lo_iou,*) '... reading unitcell and supercell'
  call uc%readfromfile('infile.ucposcar', verbosity=opts%verbosity)
  call uc%classify('wedge', timereversal=.true.)
  call ss%readfromfile('infile.ssposcar', verbosity=opts%verbosity)
  call ss%classify('supercell', uc)

  ! read MD/sampling data (positions/forces/meta/stat)
  if (mw%talk) write(lo_iou,*) '... reading infile.positions/forces/meta/stat'
  call sim%read_from_file(verbosity=max(opts%verbosity,0)+2, stride=opts%stride, mw=mw)
  call sim%remove_force_and_center_of_mass_drift()

  n_samples = 1.0_r8*sim%nt
  volume = ss%volume

  ! read IFCs from file (must exist)
  if (lo_does_file_exist('infile.forceconstant') .eqv. .false.) then
    call lo_stop_gracefully(['Missing infile.forceconstant (FC2). Run extract_forceconstants first.'], &
      1, __FILE__, __LINE__, mw%comm)
  end if
  call fc2%readfromfile(uc, 'infile.forceconstant', mem, verbosity=opts%verbosity)

  have_fc3 = lo_does_file_exist('infile.forceconstant_thirdorder')
  if (have_fc3) call fc3%readfromfile(uc, 'infile.forceconstant_thirdorder')

  have_fc4 = lo_does_file_exist('infile.forceconstant_fourthorder')
  if (have_fc4) call fc4%readfromfile(uc, 'infile.forceconstant_fourthorder')

  ! remap to supercell 
  call fc2%remap(uc, ss, fc2_ss)
  if (have_fc3) call fc3%remap(uc, ss, fc3_ss)
  if (have_fc4) call fc4%remap(uc, ss, fc4_ss)

  allocate(s0t(3,3,sim%nt), s2t(3,3,sim%nt), s3t(3,3,sim%nt), s4t(3,3,sim%nt), srt(3,3,sim%nt))
  s0t = 0.0_r8; s2t = 0.0_r8; s3t = 0.0_r8; s4t = 0.0_r8; srt = 0.0_r8

  ! ============================================================
  ! Compute compliance tensor
  ! ============================================================

  call fc2%get_elastic_constants(uc, mw, opts%verbosity)

  Svoigt = fc2%elastic_compliances_voigt / lo_pressure_HartreeBohr_to_GPa
  
  ! Write to file
    open(newunit=iu, file='outfile.compliance_tensor', status='replace')
    write(iu,*) '# Compliance tensor in Voigt notation (1/GPa)'
    do i = 1, 6
        write(iu,'(6ES20.10)') Svoigt(:,i)
    end do
    close(iu)
  
  ! compute per timestep on each rank, then reduce 
  do t = 1, sim%nt
    if (mod(t, mw%n) .ne. mw%r) cycle

    ! DFT stress already in sim%stat%stress (atomic units) 
    s0t(:,:,t) = sim%stat%stress(:,:,t)

    ! FC2 stress contribution
    s2loc = 0.0_r8
    do a1 = 1, ss%na
      v0 = 0.0_r8
      u1 = sim%u(:, a1, t)
      do ii = 1, fc2_ss%atom(a1)%n
        a2 = fc2_ss%atom(a1)%pair(ii)%i2
        m2 = fc2_ss%atom(a1)%pair(ii)%m
        rv2 = fc2_ss%atom(a1)%pair(ii)%r
        v0 = v0 - matmul(m2, sim%u(:, a2, t))
        do i1 = 1, 3
          do i2 = 1, 3
            do i3 = 1, 3
              s2loc(i1,i2) = s2loc(i1,i2) + (m2(i3,i1)*u1(i3)*rv2(i2)) / volume
            end do
          end do
        end do
      end do
      s2loc = s2loc - outer(u1, v0)/volume
    end do
    s2t(:,:,t) = s2loc

    ! FC3 stress contribution (quasi-harmonic) if present 
    if (have_fc3) then
      s3loc = 0.0_r8
      do a1 = 1, fc3_ss%na
        v0 = 0.0_r8
        do ii = 1, fc3_ss%atom(a1)%n
          m3 = fc3_ss%atom(a1)%triplet(ii)%m
          a2 = fc3_ss%atom(a1)%triplet(ii)%i2
          a3 = fc3_ss%atom(a1)%triplet(ii)%i3
          u1 = sim%u(:, a1, t)
          u2 = sim%u(:, a2, t)
          u3 = sim%u(:, a3, t)
          rv3 = fc3_ss%atom(a1)%triplet(ii)%rv3
          do i1=1,3
            do i2=1,3
              do i3=1,3
                v0(i1) = v0(i1) - m3(i1,i2,i3)*u2(i2)*u3(i3)
                do i4=1,3
                  s3loc(i1,i2) = s3loc(i1,i2) + 0.5_r8/volume * (m3(i3,i4,i2)*u1(i3)*u2(i4)*rv3(i1))
                end do
              end do
            end do
          end do
        end do
        v0 = v0*0.5_r8
        s3loc = s3loc - outer(sim%u(:,a1,t), v0)/volume
      end do
      s3t(:,:,t) = s3loc
    end if

    ! FC4 stress (optional)
    if (have_fc4) then
      s4loc = 0.0_r8
      do a1 = 1, fc4_ss%na
        v0 = 0.0_r8
        do ii = 1, fc4_ss%atom(a1)%n
          m4 = fc4_ss%atom(a1)%quartet(ii)%m
          a2 = fc4_ss%atom(a1)%quartet(ii)%i2
          a3 = fc4_ss%atom(a1)%quartet(ii)%i3
          a4 = fc4_ss%atom(a1)%quartet(ii)%i4
          u1 = sim%u(:, a1, t)
          u2 = sim%u(:, a2, t)
          u3 = sim%u(:, a3, t)
          u4 = sim%u(:, a4, t)
          rv4 = fc4_ss%atom(a1)%quartet(ii)%rv4
          do i1=1,3
            do i2=1,3
              do i3=1,3
                do i4=1,3
                  v0(i1) = v0(i1) - m4(i1,i2,i3,i4)*u2(i2)*u3(i3)*u4(i4)
                  do i5=1,3
                    s4loc(i1,i2) = s4loc(i1,i2) + (1.0_r8/volume/6.0_r8) * (u1(i3)*u2(i4)*u3(i5)*m4(i3,i4,i5,i1)*rv4(i2))
                  end do
                end do
              end do
            end do
          end do
        end do
      end do
      s4t(:,:,t) = s4loc
    end if

  end do

    ! reduce time series across ranks 
  call mw%allreduce('sum', s0t)
  call mw%allreduce('sum', s2t)
  call mw%allreduce('sum', s3t)
  call mw%allreduce('sum', s4t)

  ! residual in atomic units then convert to GPa 
  srt = s0t - s2t - s3t - s4t

  s0t = s0t * lo_pressure_HartreeBohr_to_GPa
  s2t = s2t * lo_pressure_HartreeBohr_to_GPa
  s3t = s3t * lo_pressure_HartreeBohr_to_GPa
  s4t = s4t * lo_pressure_HartreeBohr_to_GPa
  srt = srt * lo_pressure_HartreeBohr_to_GPa

  ! compute mean/std/err
  do i=1,3
    do j=1,3
      s0(i,j)=lo_mean(s0t(i,j,:)); s2(i,j)=lo_mean(s2t(i,j,:)); s3(i,j)=lo_mean(s3t(i,j,:))
      s4(i,j)=lo_mean(s4t(i,j,:)); sr(i,j)=lo_mean(srt(i,j,:))
      s0_std(i,j)=lo_stddev(s0t(i,j,:)); s2_std(i,j)=lo_stddev(s2t(i,j,:)); s3_std(i,j)=lo_stddev(s3t(i,j,:))
      s4_std(i,j)=lo_stddev(s4t(i,j,:)); sr_std(i,j)=lo_stddev(srt(i,j,:))
    end do
  end do
  ! diagnostic: mean and spread of (DFT - FC2) alone, from the same time series.
  ! Eq 5.26 intends the FC2 subtraction to REDUCE fluctuations (~10x); if
  ! std(s0-s2) > std(s0) the term is not acting as a swap.
  do i=1,3
    do j=1,3
      d02(i,j)     = lo_mean  (s0t(i,j,:) - s2t(i,j,:))
      d02_std(i,j) = lo_stddev(s0t(i,j,:) - s2t(i,j,:))
    end do
  end do
  d02_err = d02_std/sqrt(n_samples)

  s0_err = s0_std/sqrt(n_samples)
  s2_err = s2_std/sqrt(n_samples)
  s3_err = s3_std/sqrt(n_samples)
  s4_err = s4_std/sqrt(n_samples)
  sr_err = sr_std/sqrt(n_samples)

  ! symmetrize 
  call symmetrize_stress(s0, uc, s0_sym)
  call symmetrize_stress(s2, uc, s2_sym)
  call symmetrize_stress(s3, uc, s3_sym)
  call symmetrize_stress(s4, uc, s4_sym)
  call symmetrize_stress(sr, uc, sr_sym)

  call symmetrize_stress(s0_std, uc, s0_std_sym)
  call symmetrize_stress(s2_std, uc, s2_std_sym)
  call symmetrize_stress(s3_std, uc, s3_std_sym)
  call symmetrize_stress(s4_std, uc, s4_std_sym)
  call symmetrize_stress(sr_std, uc, sr_std_sym)

  call symmetrize_stress(s0_err, uc, s0_err_sym)
  call symmetrize_stress(s2_err, uc, s2_err_sym)
  call symmetrize_stress(s3_err, uc, s3_err_sym)
  call symmetrize_stress(s4_err, uc, s4_err_sym)
  call symmetrize_stress(sr_err, uc, sr_err_sym)

  call symmetrize_stress(d02,     uc, d02_sym)
  call symmetrize_stress(d02_std, uc, d02_std_sym)
  call symmetrize_stress(d02_err, uc, d02_err_sym)

  if (mw%talk) then
    call write_stress_voigt(s0_sym, s0_std_sym, s0_err_sym, 'DFT stress (symmetrized) [GPa]')
    call write_stress_voigt(s2_sym, s2_std_sym, s2_err_sym, 'FC2 stress (symmetrized) [GPa]')
    if (have_fc3) call write_stress_voigt(s3_sym, s3_std_sym, s3_err_sym, 'FC3 stress (symmetrized) [GPa]')
    call write_stress_voigt(sr_sym, sr_std_sym, sr_err_sym, 'Residual stress (symmetrized) [GPa]')
  end if

  ! write only the non-CSV outfile.stress_* files 
  call write_file_stress_voigt(s0_sym, s0_std_sym, s0_err_sym, 'outfile.stress_dft', '# DFT stress (symmetrized), unit: GPa')
  call write_file_stress_voigt(s2_sym, s2_std_sym, s2_err_sym, 'outfile.stress_fc2', '# FC2 stress (symmetrized), unit: GPa')
  if (have_fc3) call write_file_stress_voigt(s3_sym, s3_std_sym, s3_err_sym, 'outfile.stress_fc3', '# FC3 stress (symmetrized), unit: GPa')
  call write_file_stress_voigt(sr_sym, sr_std_sym, sr_err_sym, 'outfile.stress_res', '# Residual stress (DFT-FC) (symmetrized), unit: GPa')
  call write_file_stress_voigt(d02_sym, d02_std_sym, d02_err_sym, 'outfile.stress_dft_minus_fc2', &
       '# DFT-FC2 stress (symmetrized), unit: GPa')

  if (mw%talk) then
    write(*,*) ''
    write(*,*) 'Done in ', tochar(walltime()-t0), ' s'
  end if
  call mw%destroy()

contains
  pure function outer(a,b) result(m)
    real(r8), intent(in) :: a(3), b(3)
    real(r8) :: m(3,3)
    integer :: i,j
    do i=1,3
      do j=1,3
        m(i,j) = a(i)*b(j)
      end do
    end do
  end function outer

end program ifc_pressure
