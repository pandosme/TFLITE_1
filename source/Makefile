PROG1	= inference
OBJS1	= main.c imgconverter.c imgprovider.c imgutils.c cJSON.c HTTP.c FILE.c APP.c STATUS.c DEVICE.c TeachableMachine.c
PROGS	= $(PROG1)

PKGS = gio-2.0 gio-2.0 gio-unix-2.0 vdostream liblarod axhttp

CFLAGS  += -Iinclude

LDFLAGS += -L./lib -Wl,-rpath,'$$ORIGIN/lib'

CFLAGS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags $(PKGS))
LDLIBS += $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs $(PKGS))
LDLIBS  += -s -lm -ldl -lyuv -ljpeg -laxparameter

CFLAGS += -DLAROD_API_VERSION_1

all:	$(PROGS)

$(PROG1): $(OBJS1)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	rm -f $(PROGS) *.o *.eap lib/* include/* model/* label/*
