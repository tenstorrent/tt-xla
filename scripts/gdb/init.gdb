printf "Executing init.gdb ...\n"

set logging file gdb.log
printf "Logging set to file gdb.log.\n"

set logging overwrite on
printf "Logging overwrite enabled.\n"

set logging redirect off
printf "Loging redirect turned off.\n"

set logging enabled on
printf "Logging enabled.\n"

set print thread-events off
printf "Thread events turned off.\n"

set pagination off
printf "Pagination turned off.\n"

set print pretty on
printf "Pretty printing enabled.\n"

python
import sys
sys.path.insert(0, "/usr/share/gcc/python")
from libstdcxx.v6.printers import register_libstdcxx_printers
register_libstdcxx_printers(None)
end
printf "Pretty printing for C++ STL enabled.\n"
