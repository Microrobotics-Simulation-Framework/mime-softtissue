"""The general part (a candidate for MIME's core later): a soft solid at large strain with a regularised fracture model
on a fixed Cartesian grid, quasi-static, in pure JAX, with no MIME import. By concern: ``material`` (the energy law,
in invariants), ``kinematics`` (one build per dimension; plane strain today), ``fracture`` (phase field, deletion
control), ``solvers`` (block-tridiagonal solve, Newton), ``specimen`` (a specimen is data; two factories), ``solver``
and ``loadpath`` (the staggered step and the load steps), ``readouts`` (implied toughness, J-integral)."""
