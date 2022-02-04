## A Problem in Logistic
Imagine you are operating a local port attached to a big company. Everyday you have a number of products that you need to ship to a train station to be carried away. You have a ship which has a limit on the maximum amount of the load it can carry, and this maximum load is smaller than the total weight of your items. So you have to load the ship, send it back and forth between your port and the train station. The goal of the project is to train an agent, which maximizes the amount of profit you can earn by arranging items in different shipping trips.

#### Items differ from each other:
- each item earns you some money for shipping it (you earn this). This could be coming from a continues or discrte random dist. For example, sending a pack of milk, earn you 1$, and sending a pack of coal 0.5$.
- each item has a weight associated with it, e.g. a pack of milk is 1kg, and a pack of coal is 10kg,
- each item can have a specific delivery time, if you deliver before that it is fine, if you deliver after it you have pay delay fees.
- the delay for each item costs you differently, e.g. delay/hour for milk is 0.01$/h for the first day, and 0.5$ for the later days (as it expires!). Coal is 0.001$/h for as long as you want.

#### Your profit is a combination of:
- the money you earn for shipping each item,
- the money you loose per delay of each item,
- fuel cost for shipping which depends on the ship's load , e.g. sending an empty ship on a trip is cheaper than a fully loaded one.

Now, again, **the goal is to choose packages for each trip, such that you have the maximum profit at the end of the day.**


#### The more complicated version is
**Knapsack Problem (KS)**, as the agent has to plan for many times filling in the ship, and the order of the shipping matters. KS problem is one of the most famous problems in Operation Research (OR):
https://en.wikipedia.org/wiki/Knapsack_problem


#### Possible extensions
- Having more than one destination (let's say ports A, B, ...) , but still having one ship. Here the destinations are independent of each other, i.e. to get your product to port B you dont need to pass it through port A or so. Here the ship should also plan what to take to which destination first, or can it plan a round trip to visit all the ports.
- Multi-agent extension would be having more than one destination, and more than one ship [I have to think how much multi agency is really here, they should definitely cooperate but whther this can be always easily reformulated as two separate agents or not.]
- Having  more than one destination in a hierarchical fashion: you deliver to A (which is a major station), from A the products should go to A_1 and A_2 (which are smaller than A). You can think of shipping from China to a port in NY, from NY to smaller ports along the coast or by trucks to different cities. One should consider that the carries between the A and A_i are generally smaller in their load capacity, e.g what is shipped to the US by huge ships is finally distributed by UPS trucks.


#### Challenges and Advantages
- First and foremost, it is not yet another game!
- Second, this problem has a very cheap (computationally) environment, which is in favor of your learning as you can easily try different algorithms and solution schemes
- Third, it can have a very large action space, which is a very nice technical challenge, and also very much in attention of the community.
- Forth, it is not far from a real application. One can easily build upon its solution many additional levels of complication. As soon as you have a working solution the door is opened for the next level.


## Environment
1. `conda create -n logistics_w_RL_v1 python=3.8`
2. `conda activate logistics_w_RL_v1`
3. `pip install --upgrade pip`
4. `pip install -r requirements_logictics_w_RL.txt`