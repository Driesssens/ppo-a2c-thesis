from gym_minigrid.minigrid import *


class SmallUnlock8x8(MiniGridEnv):

    def __init__(self, size=8):
        super().__init__(
            grid_size=size,
            max_steps=8 * size * size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect(1, 1, width-2, height-2)

        # Place the agent in the top-left corner
        self.start_pos = (2, 2)
        self.start_dir = 0

        self.key = Key()
        door_y = self._rand_int(3, height-2)
        self.place_obj(self.key)
        self.door = LockedDoor(color=self.key.color)
        self.grid.set(width-2, door_y, self.door)
        self.picked_up_key = False

        self.mission = ""

    def step(self, action):
        obs, reward, done, info = super().step(action)

        if not self.picked_up_key and action == self.actions.pickup:
            if self.carrying and self.carrying.type == self.key.type:
                reward = 0 * self._reward()
                self.picked_up_key = True

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = 1 * self._reward()
                done = True

        return obs, reward, done, info
