from __future__ import division, print_function

import sys
import numpy as np
import gym
import time
from optparse import OptionParser

import gym_minigrid
from gym_minigrid.minigrid import WorldObj, MiniGridEnv, Lava, Grid, Wall, Goal
from gym_minigrid.envs.crossing import CrossingEnv
# from minigrid_utils import HardSubgoalCrossingEnv, HardEuclidRewardCrossingEnv, HardOptRewardCrossingEnv, \
#                                 DeterministicCrossingEnv, DetHardOptRewardCrossingEnv

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)
#     env = HardSubgoalCrossingEnv()
#     env = HardEuclidRewardCrossingEnv()
#     env = HardOptRewardCrossingEnv()
#     env = DeterministicCrossingEnv()
#     env = DetHardOptRewardCrossingEnv()

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        # Screenshot funcitonality
        elif keyName == 'ALT':
            screen_path = options.env_name + '.png'
            print('saving screenshot "{}"'.format(screen_path))
            pixmap = env.render('pixmap')
            pixmap.save(screen_path)
            return

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)

        print('step=%s, reward=%.2f, ' % (env.step_count, reward), info)

        if done:
            print('done!')
            resetEnv()

        def rolling_window(a, window):
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            x = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
            return x
    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break
# je rajoute ca pour corriger le bug ndarray has no window
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
if __name__ == "__main__":
    main()