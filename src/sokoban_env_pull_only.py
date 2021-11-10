from gym_sokoban.envs.sokoban_env import SokobanEnv, CHANGE_COORDINATES

from gym.spaces import Box
from gym.spaces.discrete import Discrete
from gym_sokoban.envs.room_utils import generate_room

import numpy as np


class PullSokobanEnv(SokobanEnv):

    def __init__(self,
                 dim_room=(10, 10),
                 max_steps=120,
                 num_boxes=3,
                 num_gen_steps=None):

        super(PullSokobanEnv, self).__init__(dim_room, max_steps, num_boxes, num_gen_steps)
        screen_height, screen_width = (dim_room[0] * 16, dim_room[1] * 16)
        self.observation_space = Box(low=0, high=255, shape=(screen_height, screen_width, 3))
        self.boxes_are_on_target = [False] * num_boxes
        self.action_space = Discrete(len(ACTION_LOOKUP))

        # Penalties and Rewards for reverse Sokoban
        self.penalty_for_step = -0.1
        self.penalty_box_off_target = 1
        self.reward_box_on_target = -1
        self.reward_finished = 10
        self.reward_last = 0

        _ = self.reset()


    # Overwrite parent _push function
    def _push(self, action):
        raise Exception("Can’t use push action in this mode")

    # Actually OFF target, not on
    def _check_if_all_boxes_on_target(self):
        are_all_boxes_off_targets = self.boxes_on_target == 0
        return are_all_boxes_off_targets

    def reset(self, second_player=False, render_mode='rgb_array'):
        try:
            self.room_fixed, self.room_state, self.box_mapping = generate_room(
                dim=self.dim_room,
                num_steps=self.num_gen_steps,
                num_boxes=self.num_boxes,
                second_player=second_player
            )
        except (RuntimeError, RuntimeWarning) as e:
            print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
            print("[SOKOBAN] Retry . . .")
            return self.reset(second_player=second_player, render_mode=render_mode)

        self.player_position = np.argwhere(self.room_state == 5)[0]
        self.num_env_steps = 0
        self.reward_last = 0
        self.boxes_on_target = self.num_boxes

        # Change target into boxes on target and boxes into empty space
        self.room_state[self.room_state == 2] = 3
        self.room_state[self.room_state == 4] = 1

        starting_observation = self.render(render_mode)
        return starting_observation

    def step(self, action, observation_mode='rgb_array'):
        assert action in ACTION_LOOKUP

        self.num_env_steps += 1

        self.new_box_position = None
        self.old_box_position = None

        moved_box = False
        if action == 0:
            moved_player = False

        # All pull actions are in the range of [0, 3]
        if action < 5:
            moved_player, moved_box = self._pull(action)

        else:
            moved_player = self._move(action)

        self._calc_reward()

        done = self._check_if_done()

        # Convert the observation to RGB frame
        observation = self.render(mode=observation_mode)

        info = {
            "action.name": ACTION_LOOKUP[action],
            "action.moved_player": moved_player,
            "action.moved_box": moved_box,
        }
        if done:
            info["maxsteps_used"] = self._check_if_maxsteps()
            info["all_boxes_on_target"] = self._check_if_all_boxes_on_target()

        return observation, self.reward_last, done, info

    def _pull(self, action):
        """
        Moves the player to the next field, if it is not occupied.
        :param action:
        :return: Boolean, indicating a change of the room's state
        """
        change = CHANGE_COORDINATES[(action - 1) % 4]
        new_position = self.player_position + change
        current_position = self.player_position.copy()
        pull_content_position = self.player_position - change

        # Move player if the field in the moving direction is either
        # an empty field or an empty box target.
        if self.room_state[new_position[0], new_position[1]] in [1, 2]:
            self.player_position = new_position
            self.room_state[(new_position[0], new_position[1])] = 5
            self.room_state[current_position[0], current_position[1]] = \
                self.room_fixed[current_position[0], current_position[1]]

            box_next_to_player = self.room_state[pull_content_position[0], pull_content_position[1]] in [3, 4]
            if box_next_to_player:
                # Move Box
                box_type = 4
                if self.room_fixed[current_position[0], current_position[1]] == 2:
                    box_type = 3
                self.room_state[current_position[0], current_position[1]] = box_type
                self.room_state[pull_content_position[0], pull_content_position[1]] = \
                    self.room_fixed[pull_content_position[0], pull_content_position[1]]

            return True, box_next_to_player

        return False, False

    def get_action_lookup(self):
        return ACTION_LOOKUP

    def get_action_meanings(self):
        return ACTION_LOOKUP


ACTION_LOOKUP = {
    0: 'no operation',
    1: 'pull up',
    2: 'pull down',
    3: 'pull left',
    4: 'pull right',
    5: 'move up',
    6: 'move down',
    7: 'move left',
    8: 'move right',
}
