# FlappyBird
 FlappyBird reinforcement learning project

results after 3 hours:

![](https://i.imgur.com/M51TCHp.gif)


"dqnmodel.h5" is the already trained model. Run "FBirdTS2Teste.py" to see it working!

to train the model, edit the hyperparameters in the "FBirdTS2.py" file and run it.


About the project:

The goal of this project was to get a some-what-working flappy bird agent. As you can see in the gif above, the goal was accomplished. To train the model, I first edited the gym-ple 'FlappyBird-v0' environment so that the pipes would have the gap size doubled. after the model quickly converged in that edited environment, i moved on to the default pipe size. Also, to deal with the sparce rewards problem, it was needed to adapt the rewards.

By Pedro Rosa
