JULIA IS COLUMN-MAJOR!!
A[row,col] is fine, but A[3] == A[1, 2] for a 2x2 matrix

A = copy(B) to avoid aliasing

reverse(x) doesn't actually modify x, but reverse!(x) does

all ones() and zeroes() create float values by default
to create integer, use ones(Int8, 4,4) to create 4x4

using <package> to import packages

using SparseArrays
B = spzeros(100,100)  # creates sparse array
B_ = view(B, 4:10, 50:60)
B_[5, 5] = 10         # modifies B[4 + 5, 5 + 5] = 10, also note 1-indexing, not 0-indexing

Diagonal(A)           # forms diagonal matrix using A's diagonals
Diagonal([1,2,3])     # creates diagonal with 1,2,3

LowerTriangular(A)    # creates an actual "Lower Triangular" type which is useful when performing inverse, or other computation since compiler can use more optimal techniques


**** SUPER USEFUL FOR OUR PROJECT ****
issymmetric(A)
isposdef(A)
rank(A)

@time A\b             # can time how long an operation takes

e = eigen(A)
e.values
e.vectors


Julia caches results and operations:
julia> @time mysum(x)
  0.013456 seconds (8.93 k allocations: 444.650 KiB)
498.3609213304516

julia> @time mysum(x)
  0.000005 seconds (1 allocation: 16 bytes)
498.3609213304516

julia> function mysum(x)
           out = zero(eltype(x))
           for a in x
               out += a
           end
           out
       end

notice for loop is same as python except no colon
notice to return out, you need to write "out" at the end
notice you can apply this to any object type, eltype will handle for you

function myfunc(a,b,c=10;verbose=true)
c is a normal arg with default value
verbose is a kwarg, so to specify, you need to explicitly state verbose=false
myfunc(1,2)
myfunc(1,2,3)  # c = 3
myfunc(1,2,3,false) XXX WRONG
myfunc(1,2,3, verbose=false)


julia> function bar(x...)
           println(length(x))
           end
bar (generic function with 1 method)

julia> bar(1,2,3)
3

julia> bar(1,2)
2

julia> t = (1,2,3)
(1, 2, 3)

julia> bar(t...)
3

julia> bar(t)
1


Notice that to unpack, you use ...
to unpack t, need to explicitly use ...
bar can take in any number of arguments


similar to C++, can define various function templates for the same function
foo(x,y) = println("this is the generic function")
foo(1,2)
foo(1,"a")
foo(x::Integer, y::Integer) = println("both ints")
f(1,2)         # both ints
foo(1, 2.123)  # this is a generic function



A' is A transpose, there is no .T

norm(A), norm(A, 1), norm(A, p)  # perform differnt p-norm operations, default to 2

A\B                   # solves Ax = b... if you specify type of A (ex: symmetric, lower triangular, etc... this can boost performance since compiler knows optimal method to use)


struct Foo
           bar
           baz::Int
           qux::Float64
end

# the above entries are not mutable, so need to add mutable struct for that behavior
