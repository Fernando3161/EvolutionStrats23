
# Chapter 1
    # Solution Space
    # Evolutionary cycle
    # Exploration and Exploitation
    # Sphere
        # A typical numerical test function is the Sphere function f (x) = xT · x
    # Selection
    # Mutation operator
    # Mutation operator characterisitcs
        # Reachability
        # Unbiasedness
        # Scalability
        # typical mutation operator is Gaussian mutation: x′ = x + σ · N (0, 1)

# Chapter 2
    # Intermediate Recombination
        # Crossover combines two or more randomly chosen parents,
        # typically using the arithmetic mean of ρ parents:
        # 1/p sum(x_i)i->p
        # p = parent, P = parents population
    # In case of multi-recombination, i.e. ρ = µ, also written as
        # (µ/µ + λ)-ES, the arithmetic mean of all solutions from the
        # parental population P is computed

    # Dominant Recombination
        # An alternative to intermediate is dominant recombination that
        # randomly selects component xi from one of the ρ parents:
        # (x)i = (xj)i with j = random {1, . . . , ρ}

    #Plus Selection (µ + λ)-ES
        # selects the µ best solutions from the λ offspring and the µ parents

    # Comma Selection (µ, λ)-ES
        # selects only from the λ offspring. It forgets the parental solutions,
        # which may allow to leave local ptima

    # Test Functions
        # Sphere
            # sum(x^2)i->N
        # Tangent
            # -((sum(x)i->N)-N <= 0
        # Doublesum
            # sum(x)j->i
        # Rosenbrock
        # Rastrigin
        # OneMax
        # Wilcoxon test ? 0.05 is the threshold for significance

# Chapter 3
    # Rechenberg
        # Parameters tuning and adaption is important
            # Controll
                #self-adaptive (slide 26)
                # dynamic adaption
            # Tuning
                # manual tuning
                # DoE Design of Exeptive
        # Rule: The If s/g > 3/5,   is increased with:
        #   sigma =   sigma · tau (6)
        # not changed in case of s/g = 1/5 and decreased otherwise with:
        #   sigma :=  sigma/tau (7)
        # with tau > 1

# Chapter 4 Rechenberg
    #


# Chapter 5 Meta-Evolution
    #

# Chapter 6 Restart
    # Numerous mechanisms aim at overcoming local optima like
    # large populations, comma selection, and ﬁtness proportional
    # selection

    # Restarting is a strategy that restarts numerous ES runs from
    # uniformly random starting points

    # The restart mechanism can be combined with convergence
    # criteria for termination, i.e., a run is terminated, if the ﬁtness
    # improvement of the best solutions falls below a small threshold
    # ✏ > 0:

    # Instead of uniformly random starting points, Gaussian noise with
    # strengths  ˜ can be added to the stuck population, i.e. to the
    # starting point x0


# Chapter 7: Self-Adaptation (SA)
    # Name (1, lamda)-sigma-SA-ES
    # good sigmas lead to good solutions vice versa
    # solutions and sigmas are selected by fitness of the solution
    # solutions and sigmas are crossovered (intermediate recombination)

    # stepzsize adaptation is the first step
    # scalar is the second step
    # the scalar can be individual for each dimension! = multivariate

# Chapter 8: Evolution Path
    # Evolution path cumulates the mutation directions
    # selection process by fitness and evolution path
    # divided by the expected length of the normal distributin (0=?)
    # if the equation part in brackets of eq. 10 becomes negative the sigma?
    # "the smaller the s_o the smaller the stepsize"

    # draw these numbers and prove that this is true (eq. 11-15?) page 60 prove given statements