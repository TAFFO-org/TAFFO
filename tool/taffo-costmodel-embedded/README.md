# taffo-costmodel-embedded

This tool is used to compute TAFFO DTA ILP models for STMicroelectronics
ARM-based embedded evaluation boards.

## How to use it

**Step 1**: Install a cross-compilation ARM toolchain. This tool is tested
with the official GCC toolchain distributed by ARM, freely downloadable here:
https://developer.arm.com/tools-and-software/open-source-software/developer-tools/gnu-toolchain/downloads

Be careful when installing cross-compilers using package managers on Linux
as there might be different package for hard-float and soft-float compilers.
Which one you need to use depends on the target.

Additionally, install the open source st-link toolkit via your package
manager of choice (https://github.com/stlink-org/stlink).

**Step 2**: Compile a flash image for your target platform. To do so, just run
a `make` command like this:

    $ make TARGET=stm32f207

where `stm32f207` can be replaced with any of the supported platform identifiers
(see later in this Readme for a list).

If everything goes well, this command will produce a file named
`taffo-costmodel.bin`. The makefile assumes the GNU ARM toolchain is in the
current PATH, so modify the PATH to allow the compiler to be found.

**Step 3**: Connect the board via the USB St-Link connection and flash the chip
with the following command:

    $ make flash

Do not connect multiple boards at the same time, as the makefile
does not provide enough information for the programmer frontend to distinguish
which board is the right one.

On boards with multiple USB ports connect only the port of the St-Link
interface. St-Link is a shitty protocol and if you connect the other ports as
well the tool might not be able to distinguish what is a real St-Link port
and what is not.

**Step 4**: Connect the device USB or serial port and read the output from it.

The makefile provides a phony target for this called `monitor`, but you'll need
to check in `/dev` for the actual USB device name of your serial adapter or
virtual serial port and modify the `Makefile.in` file to update it.

## Supported boards

The following boards are currently supported:

### Target ID `stm32f207`

STMicroelectronics STM3220G-EVAL board (STM32F207 SOC).

Use the RS232 DB-9 serial port already available on the device. The ST-Link
port does NOT expose any builtin serial to USB adapter!

### Target ID `stm32f407`

STMicroelectronics STM32F4-discovery board (STM32F407 SOC).

Use the virtual serial port on the device-side USB connection. To allow enough
time for connecting to the port, the software waits 10 seconds before actually
booting and starting the benchmarks.
