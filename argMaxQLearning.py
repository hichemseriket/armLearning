# # q-learning example
# # https://en.wikipedia.org/wiki/Q-learning
#
# # import matrix
# # import tgt
# import numpy
#
# # 2x3 grid world
# # S for starting grid G for goal/terminal grid
# # actions left right down up
# #  4 5 6 state
# #########
# # [0,0,G]
# # [S,0,0]
# #########
# #  1 2 3 state
#
# # setting seed
# set.seed(2016)
# # number of iterations
# N = 10
# # discount factor
# gamma = 0.9
#
# # learning rate
# alpha = 0.1
#
# # target state
# tgt.state = 6
# # Reward and action-value matrix
# Row = state(1:6)
# Column = actions(1:4)[Left, Right, Down, Up in that order]
# # reward matrix starting grid has -0.1 and ending grid has 1
# R = matrix(c(NA, 0, NA, 0,
#              -0.1, 0, NA, 0,
#              0, NA, NA, 1,
#              NA, 0, -0.1, NA,
#              0, 1, 0, NA,
#              0, NA, 0, NA
#              ),
#            nrow=6, ncol=4, byrow=TRUE)
#
# # initializing Q matrix with zeros
# Q = matrix(rep(0, len=dim(R)[1] * dim(R)[2]), nrow=dim(R)[1], ncol=dim(R)[2])
#
# for (i in 1:N) {
#     ## for each episode, choose an initial state at random
#     cs < - 1
# ## iterate until we get to the tgt.state
#     while (1) {
#     ## choose next state from possible actions at current state
#     ## Note: if only one possible action, then choose it;
#     ## otherwise, choose one at random
#     next.states < - which(R[cs, ] > -1)
#     if (length(next.states) == 1)
#     ns < - next.states
#     else
#     ns < - sample(next.states, 1)
#     ## this is the update
#     Q[cs, ns] < - Q[cs, ns] + alpha * (R[cs, ns] + gamma * max(Q[ns, which(R[ns, ] > -1)]) - Q[cs, ns])
#     ## break out of while loop if target state is reached
#     ## otherwise, set next.state as current.state and repeat
#     if (ns == tgt.state)
# break
# cs < - ns
# Sys.sleep(0.5)
# print(Q)
# }
# }
