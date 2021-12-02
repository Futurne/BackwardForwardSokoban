# BackwardForwardSokoban
Implementation of the paper "Solving Sokoban with Forward-backward Reinforcement Learning" by Y. Shoham &amp; G. Elidan

Install `gym-sokoban` from [source](https://github.com/mpSchrader/gym-sokoban).
The `raw` representation of the states is only available on the latest versions.


TODO:
* Handle deadlock backward levels (with connectivity issue)
* When does the backprop is made ?
If it is made at the end of an episode
=> don't update all nodes based on new model prediction (which doesn't change)
* Organize the unittests and experiments more clearily
