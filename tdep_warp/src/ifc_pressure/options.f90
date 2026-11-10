#include "precompilerdefinitions"
module options
  use konstanter, only: flyt, lo_status, lo_author, lo_version, lo_licence, lo_hugeint
  use flap, only: command_line_interface
  implicit none
  private
  public :: lo_opts

  type lo_opts
    integer :: verbosity = 0
    integer :: stride = 1
  contains
    procedure :: parse
  end type

contains

  subroutine parse(opts)
    class(lo_opts), intent(out) :: opts
    type(command_line_interface) :: cli
    logical :: dumlog

    call cli%init(progname='ifc_pressure', &
      authors=lo_author, version=lo_version, license=lo_licence, &
      help='Usage: ', &
      description='Compute (symmetrized) DFT/FC2/FC3/residual stresses from infile.* and infile.forceconstant* and write outfile.stress_* files.')

    call cli%add(switch='--stride', switch_ab='-s', &
      help='Use every N configuration instead of all.', required=.false., act='store', def='1', error=lo_status)
    if (lo_status /= 0) stop

    call cli%add(switch='--verbose', help='Verbose output on rank 0', required=.false., act='store_true', def='.false.', error=lo_status)
    if (lo_status /= 0) stop

    call cli%add(switch='--manpage', hidden=.true., help='', required=.false., act='store_true', def='.false.', error=lo_status)
    if (lo_status /= 0) stop

    call cli%parse(error=lo_status)
    if (lo_status /= 0) stop

    call cli%get(switch='--manpage', val=dumlog)
    if (dumlog) then
      call cli%save_man_page(trim(cli%progname)//'.1')
      call cli%save_usage_to_markdown(trim(cli%progname)//'.md')
      write(*,*) 'Wrote manpage for "'//trim(cli%progname)//'"'
      stop
    end if

    opts%verbosity = 0
    call cli%get(switch='--verbose', val=dumlog)
    if (dumlog) opts%verbosity = 1

    call cli%get(switch='--stride', val=opts%stride)
  end subroutine parse

end module options
