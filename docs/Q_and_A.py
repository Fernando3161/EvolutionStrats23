# I collect the questions and ansers asked/given by Oliver
"""
Monday
Q: How to differentiate min or max problems?
A: > or <

Q: Is „sigma“ scalable?
A. Yes

Restart?
Rechenberg Rule?

Tuesday
Q: Why Gaussian and not uniform?
A: Uniform would also work but does not allow a smooth adaptation and conversion. Test it by changing the distributions. In 1+1 it is not that easy to notice. As soon as we have parameter control, it is quiet more noticeable.

Q: Why mutation at all?
A: Just recombination is limited.



Wednesday
Q: Why sigma decrasing?
A: Small sigma at the start increases the change of being stuck in a local optimum. Decreasing over time increases the chance of getting closer to the optimum.
First its easier to get better result and in the end you need smaller sigma to find the best narrow optimum? Chance of finding the optima decreases if we keep sigma constant

Q: What happens if we adapt mhu or lambda?
A: I would say, if we decrease the parents (mhu), then we have a higher chance of offsprings with a fitness closer to the parents. But if the parents are stuck in the local optima...
If we increase the children we have a higher change to get out of local optima since we have more mutated solutions
There are not many approaches who adapt population. It makes sense if we have a lot of local optima. Exploration vs exploitation! First Explore the whole solution space, then we want to exploit the most promising. The ration “successfullness” depends on the solution area size of the fitness function. But most of the time we don’t know the local optima.
	Problems with similar fitness functions are solved with similar algorithm parameters

Rechenberg rule: The more successful we are, the bigger our stepsize can be

Q: What is the complexity of sorting?
A: N log(N)

Q: How often should we sort for mu?
A: N * mu


Choosing the right sorting operator: 1. Nexts.sort or 2.  For i in next: take smallest
Sorting the best: N * mu
When mu is smaller then log n (parents smaller log(children))
Chapter 6:

Chapter 7: Self-Adaptation
Q: Difference between Rechenberg and SA.
A: Rechenberg variates the sigma by the number of successes. SA variates the sigma by the fitness of the solution.

Thursday
Chapter 9: Covariance Matrix Estimatio

Chapter: Constraints

Q: How many loops of Algorithms are typically used?
A1: Literature gives typical choices for similar problems. Adaption afterwards - Parametrisation.
    -->Leads to own neural network (brain) = Experience
A2: Automize the parametrisation with an AI.
A3: 80 % of problems can be solved with "standard" Parametrisation.

Repair function
Q: Hopw to find the constraint boundary line
A: Split the line by half. Check middle. Repeat
A: OR: Measure penalty miniimize penalty

Friday




"""