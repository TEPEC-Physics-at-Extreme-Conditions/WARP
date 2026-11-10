include Makefile.inc
.DEFAULT_GOAL := all

CODE = ifc_pressure
PROG = ../../bin/$(CODE)
OBJECT_PATH = ../../build/$(CODE)

OBJS = $(OBJECT_PATH)/main.o \
       $(OBJECT_PATH)/options.o \
       $(OBJECT_PATH)/symmetrize.o \
       $(OBJECT_PATH)/helper.o

LPATH = -L../../lib $(blaslapackLPATH) $(incLPATHhdf) $(incLPATHmpi)
IPATH = -I../../inc/libolle -I../../inc/libflap $(blaslapackIPATH) $(incIPATHhdf) $(incIPATHmpi)
LIBS  = ../../lib/libolle.a -lflap $(blaslapackLIBS) $(incLIBShdf) $(incLIBSmpi)

F90 = $(FC) $(LPATH) $(IPATH) $(MODULE_FLAG) $(OBJECT_PATH)
F90FLAGS = $(OPT) $(MODS) $(LIBS)

# auto-create object/mod directory:
$(OBJECT_PATH):
	@mkdir -p $(OBJECT_PATH)

all: $(PROG)

$(PROG): $(OBJS)
	$(F90) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	rm -f $(PROG)
	rm -f $(OBJS) $(OBJECT_PATH)/*.mod

# --- Step 1 fix: build module-providers first, then main ---
$(OBJECT_PATH)/main.o: $(OBJECT_PATH)/options.o $(OBJECT_PATH)/symmetrize.o $(OBJECT_PATH)/helper.o | $(OBJECT_PATH)
	$(F90) $(F90FLAGS) -c main.f90 -o $@

$(OBJECT_PATH)/options.o: | $(OBJECT_PATH)
	$(F90) $(F90FLAGS) -c options.f90 -o $@

$(OBJECT_PATH)/symmetrize.o: | $(OBJECT_PATH)
	$(F90) $(F90FLAGS) -c symmetrize.f90 -o $@

$(OBJECT_PATH)/helper.o: | $(OBJECT_PATH)
	$(F90) $(F90FLAGS) -c helper.f90 -o $@
