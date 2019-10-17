var documenterSearchIndex = {"docs":
[{"location":"redirect/#Animations-1","page":"Animations","title":"Animations","text":"","category":"section"},{"location":"redirect/#","page":"Animations","title":"Animations","text":"<meta http-equiv=\"refresh\" content=\"0;URL=./optimization.html\">","category":"page"},{"location":"redirect/#","page":"Animations","title":"Animations","text":"If you are not redirected automatically, follow the link.","category":"page"},{"location":"#AnimatedOptimization.jl-1","page":"Home","title":"AnimatedOptimization.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Some optimization algorithms (for any function) with animations for functions from R² → R.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"These are meant for teaching. They have not been extenisvely tested and are likely not well-suited for other uses. ","category":"page"},{"location":"#","page":"Home","title":"Home","text":"For usage see this document or a notebook version.","category":"page"},{"location":"#API-1","page":"Home","title":"API","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Modules = [AnimatedOptimization]","category":"page"},{"location":"#AnimatedOptimization.graddescent-Tuple{Any,Any}","page":"Home","title":"AnimatedOptimization.graddescent","text":" graddescent(f, x0; grad=x->Forwardiff.gradient(f,x),\n             γ0=1.0, ftol = 1e-6,\n             xtol = 1e-4, gtol=1-6, maxiter = 1000, \n             xrange=[-2., 3.],\n             yrange=[-2.,6.], animate=true)\n\nFind the minimum of function f by gradient descent\n\nArguments\n\nf function to minimize\nx0 starting value\ngrad function that computes gradient of f\nγ0 initial step size multiplier\nftol tolerance for function value\nxtol tolerance for x\ngtol tolerance for gradient. Convergence requires meeting all three tolerances.\nmaxiter maximum iterations\nxrange x-axis range for animation\nyrange y-axis range for animation\nanimate whether to create animation\n\nReturns\n\n(fmin, xmin, iter, info, anim) tuple consisting of minimal function value, minimizer, number of iterations, convergence info, and animations\n\n\n\n\n\n","category":"method"},{"location":"#AnimatedOptimization.interiorpoint-Tuple{Any,Any,Any}","page":"Home","title":"AnimatedOptimization.interiorpoint","text":"interiorpoint(f, x0, c; \n              L   = (x,λ)->(f(x) - dot(λ,c(x))),\n              ∇ₓL = (x,λ)->ForwardDiff.gradient(z->L(z,λ), x),\n              ∇²ₓL= (x,λ)->ForwardDiff.hessian(z->L(z,λ), x),\n              ∇c = x->ForwardDiff.jacobian(c,x),\n              tol=1e-4, maxiter = 1000,\n              μ0 = 1.0, μfactor = 0.2,\n              xrange=[-2., 3.],\n              yrange=[-2.,6.], animate=true)\n\nFind the minimum of function f subject to c(x) >= 0 using a primal-dual interior point method.\n\nArguments\n\nf function to minimizie\nx0 starting value. Must have c(x0) > 0\nc constraint function. Must return an array.\nL   = (x,λ)->(f(x) - dot(λ,c(x))) Lagrangian\n∇ₓL = (x,λ)->ForwardDiff.gradient(z->L(z,λ), x) Derivative of Lagrangian wrt x\n∇²ₓL= (x,λ)->ForwardDiff.hessian(z->L(z,λ), x) Hessian of Lagrangian wrt x\n∇c = x->ForwardDiff.jacobian(c,x) Jacobian of constraints\ntol convergence tolerance\nμ0 initial μ\nμfactor how much to decrease μ by\nxrange range of x-axis for animation\nyrange range of y-axis for animation\nanimate whether to create an animation (if true requires length(x)==2)\nverbosity higher values result in more printed output during search. 0 for no output, any number > 0 for some.  \n\nReturns\n\n(fmin, xmin, iter, info, animate) tuple consisting of minimal function value, minimizer, number of iterations, and convergence info\n\n\n\n\n\n","category":"method"},{"location":"#AnimatedOptimization.minrandomsearch-Tuple{Any,Any,Any}","page":"Home","title":"AnimatedOptimization.minrandomsearch","text":"minrandomsearch(f, x0, npoints; var0=1.0, ftol = 1e-6,\n                     vtol = 1e-4, maxiter = 1000,\n                     vshrink=0.9, xrange=[-2., 3.],\n                     yrange=[-2.,6.])\n\nFind the minimum of function f by random search. \n\nCreates an animation illustrating search progress.\n\nArguments\n\nf function to minimizie\nx0 starting value\nnpoints number of points in cloud\nvar0 initial variance of points\nftol convergence tolerance for function value. Search terminates if both function change is less than ftol and variance is less than vtol.\nvtol convergence tolerance for variance. Search terminates if both function change is less than ftol and variance is less than vtol.\nmaxiter maximum number of iterations\nvshrink after every 100 iterations with no function improvement, the variance is reduced by this factor\nxrange range of x-axis in animation\nyrange range of y-axis in animation\nanimate whether to create animation\n\nReturns\n\n(fmin, xmin, iter, info, anim) tuple consisting of minimal function value, minimizer, number of iterations, convergence info, and an animation\n\n\n\n\n\n","category":"method"},{"location":"#AnimatedOptimization.newton-Tuple{Any,Any}","page":"Home","title":"AnimatedOptimization.newton","text":"newton(f, x0; \n       grad=x->ForwardDiff.gradient(f,x),\n       hess=x->ForwardDiff.hessian(f,x),\n       ftol = 1e-6,\n       xtol = 1e-4, gtol=1-6, maxiter = 1000, \n       xrange=[-2., 3.],\n       yrange=[-2.,6.], animate=true)\n\nFind the minimum of function f by Newton's method.\n\nArguments\n\nf function to minimizie\nx0 starting value\ngrad function that returns gradient of f\nhess function that returns hessian of f\nftol tolerance for function value\nxtol tolerance for x\ngtol tolerance for gradient. Convergence requires meeting all three tolerances.\nmaxiter maximum iterations\nxrange x-axis range for animation\nyrange y-axis range for animation\nanimate whether to create animation\n\nReturns\n\n(fmin, xmin, iter, info, anim) tuple consisting of minimal function value, minimizer, number of iterations, convergence info, and animation\n\n\n\n\n\n","category":"method"},{"location":"#AnimatedOptimization.sequentialquadratic-Tuple{Any,Any,Any}","page":"Home","title":"AnimatedOptimization.sequentialquadratic","text":"  sequentialquadratic(f, x0, c; \n                      ∇f = x->ForwardDiff.gradient(f,x),\n                      ∇c = x->ForwardDiff.jacobian(c,x),\n                      L   = (x,λ)->(f(x) - dot(λ,c(x))),\n                      ∇ₓL = (x,λ)->ForwardDiff.gradient(z->L(z,λ), x),\n                      ∇²ₓL= (x,λ)->ForwardDiff.hessian(z->L(z,λ), x),\n                      tol=1e-4, maxiter = 1000,\n                      trustradius=1.0, xrange=[-2., 3.],\n                      yrange=[-2.,6.], animate=true, verbosity=1)\n\nFind the minimum of function f by random search\n\nArguments\n\nf function to minimizie\nx0 starting value. Must have c(x0) > 0\nc constraint function. Must return an array.\n∇f = x->ForwardDiff.gradient(f,x)\n∇c = x->ForwardDiff.jacobian(c,x) Jacobian of constraints\nL   = (x,λ)->(f(x) - dot(λ,c(x))) Lagrangian\n∇ₓL = (x,λ)->ForwardDiff.gradient(z->L(z,λ), x) Derivative of Lagrangian wrt x\n∇²ₓL= (x,λ)->ForwardDiff.hessian(z->L(z,λ), x) Hessian of Lagrangian wrt x\ntol convergence tolerance\nmaxiter\ntrustradius initial trust region radius\nxrange range of x-axis for animation\nyrange range of y-axis for animation\nanimate whether to create an animation (if true requires length(x)==2)\nverbosity higher values result in more printed output during search. 0 for no output, any number > 0 for some.  \n\nReturns\n\n(fmin, xmin, iter, info, animate) tuple consisting of minimal function value, minimizer, number of iterations, and convergence info\n\n\n\n\n\n","category":"method"}]
}
