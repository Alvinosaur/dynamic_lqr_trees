Initial Template for Map visualization and square obstacles taken from this repo:
https://github.com/HiroIshida/julia_motion_planning


Holonomic RRT commit: d17db87

To run in terminal, simply call:
```
julia run_main.jl
```
This is slow though since it constantly starts up a new Julia REPL and needs to load the packages.
A faster option is to use either VSCode or Atom, start a running julia REPL in the terminal, and run the code using that existing REPL without constantly shutting it down.

In VSCode, the "Run" Button will have three options, the first called: "Julia: Execute File in REPL" 


