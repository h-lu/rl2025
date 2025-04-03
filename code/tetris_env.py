import numpy as np
import pygame
import pygame
import random
from enum import Enum
import gymnasium as gym
from gymnasium import spaces

# 方块形状定义
class TetrominoType(Enum):
    I = 0
    O = 1
    T = 2
    S = 3
    Z = 4
    J = 5
    L = 6

# 方块形状矩阵
TETROMINOES = {
    TetrominoType.I: np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ]),
    TetrominoType.O: np.array([
        [1, 1],
        [1, 1]
    ]),
    TetrominoType.T: np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 0, 0]
    ]),
    TetrominoType.S: np.array([
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 0]
    ]),
    TetrominoType.Z: np.array([
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 0]
    ]),
    TetrominoType.J: np.array([
        [1, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ]),
    TetrominoType.L: np.array([
        [0, 0, 1],
        [1, 1, 1],
        [0, 0, 0]
    ])
}

class TetrisEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # 游戏板尺寸 (20行 x 10列)
        self.board_width = 10
        self.board_height = 20
        
        # 动作空间: 0=左移, 1=右移, 2=旋转, 3=快速下落
        self.action_space = spaces.Discrete(4)
        
        # 状态空间: 游戏板(20x10) + 当前方块(7) + 下一个方块(7) + 位置(2) + 旋转(1)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(self.board_height, self.board_width), dtype=np.int8),
            "current_piece": spaces.Discrete(7),
            "next_piece": spaces.Discrete(7),
            "position": spaces.Box(low=0, high=self.board_width, shape=(2,), dtype=np.int8),
            "rotation": spaces.Discrete(4)
        })
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # 初始化游戏状态
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 清空游戏板
        self.board = np.zeros((self.board_height, self.board_width), dtype=np.int8)
        
        # 生成当前和下一个方块
        self.current_piece = self._generate_piece()
        self.next_piece = self._generate_piece()
        
        # 初始化方块位置 (顶部中间)
        self.piece_x = self.board_width // 2 - 1
        self.piece_y = 0
        self.rotation = 0
        
        # 游戏状态
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        
        # 计算初始状态
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()
        
        reward = 0
        
        # 执行动作
        if action == 0:  # 左移
            self._move(-1, 0)
        elif action == 1:  # 右移
            self._move(1, 0)
        elif action == 2:  # 旋转
            self._rotate()
        elif action == 3:  # 快速下落
            reward += self._hard_drop()
        
        # 自然下落
        if not self._move(0, 1):
            # 如果不能下落，锁定方块
            self._lock_piece()
            
            # 检查消除行
            lines = self._clear_lines()
            if lines > 0:
                reward += lines * 10  # 每消除一行+10分
                if lines == 4:
                    reward += 60  # 消除四行额外奖励
                self.lines_cleared += lines
            
            # 生成新方块
            self.current_piece = self.next_piece
            self.next_piece = self._generate_piece()
            self.piece_x = self.board_width // 2 - 1
            self.piece_y = 0
            self.rotation = 0
            
            # 检查游戏结束
            if self._check_collision():
                self.game_over = True
                reward -= 10  # 游戏结束惩罚
        
        # 存活奖励
        reward += 0.1
        
        # 计算空洞和高度差惩罚
        holes, height_diff = self._calculate_board_stats()
        reward -= holes * 0.1 + height_diff * 0.05
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, self.game_over, False, info
    
    def render(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                # 根据游戏区域实际大小设置窗口(10x20方块，每个30像素)
                self.window = pygame.display.set_mode((self.board_width * 30, self.board_height * 30))
                pygame.display.set_caption("Tetris DQN")
            if self.clock is None:
                self.clock = pygame.time.Clock()
            
            self._render_gui()
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
            self.clock = None
    
    # 内部辅助方法
    def _generate_piece(self):
        return random.choice(list(TetrominoType))
    
    def _get_observation(self):
        return {
            "board": self.board.copy(),
            "current_piece": self.current_piece.value,
            "next_piece": self.next_piece.value,
            "position": np.array([self.piece_x, self.piece_y], dtype=np.int8),
            "rotation": self.rotation
        }
    
    def _get_info(self):
        return {
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "game_over": self.game_over
        }
    
    def _get_current_piece_matrix(self):
        # 获取当前旋转状态的方块矩阵
        piece_matrix = TETROMINOES[self.current_piece]
        for _ in range(self.rotation):
            piece_matrix = np.rot90(piece_matrix)
        return piece_matrix
    
    def _check_collision(self, offset_x=0, offset_y=0):
        piece_matrix = self._get_current_piece_matrix()
        for y in range(piece_matrix.shape[0]):
            for x in range(piece_matrix.shape[1]):
                if piece_matrix[y][x]:
                    board_x = self.piece_x + x + offset_x
                    board_y = self.piece_y + y + offset_y
                    
                    if (board_x < 0 or board_x >= self.board_width or 
                        board_y >= self.board_height or 
                        (board_y >= 0 and self.board[board_y][board_x])):
                        return True
        return False
    
    def _move(self, dx, dy):
        if not self._check_collision(dx, dy):
            self.piece_x += dx
            self.piece_y += dy
            return True
        return False
    
    def _rotate(self):
        new_rotation = (self.rotation + 1) % 4
        old_rotation = self.rotation
        self.rotation = new_rotation
        
        # 如果旋转后发生碰撞，尝试左右移动
        if self._check_collision():
            if not self._move(-1, 0):
                if not self._move(1, 0):
                    self.rotation = old_rotation
    
    def _hard_drop(self):
        drop_distance = 0
        while not self._check_collision(0, 1):
            self.piece_y += 1
            drop_distance += 1
        return drop_distance * 0.1  # 下落距离奖励
    
    def _lock_piece(self):
        piece_matrix = self._get_current_piece_matrix()
        for y in range(piece_matrix.shape[0]):
            for x in range(piece_matrix.shape[1]):
                if piece_matrix[y][x]:
                    board_x = self.piece_x + x
                    board_y = self.piece_y + y
                    if 0 <= board_y < self.board_height and 0 <= board_x < self.board_width:
                        self.board[board_y][board_x] = 1
    
    def _clear_lines(self):
        lines_cleared = 0
        for y in range(self.board_height):
            if np.all(self.board[y] == 1):
                # 移除该行并添加新的空行在顶部
                self.board = np.vstack([np.zeros((1, self.board_width)), np.delete(self.board, y, 0)])
                lines_cleared += 1
        return lines_cleared
    
    def _calculate_board_stats(self):
        # 计算空洞数量和高度差
        holes = 0
        heights = np.zeros(self.board_width)
        
        for x in range(self.board_width):
            column = self.board[:, x]
            # 找到每列的最高方块
            if np.any(column == 1):
                heights[x] = self.board_height - np.argmax(column == 1)
            # 计算空洞
            if heights[x] > 0:
                holes += np.sum(column[self.board_height - int(heights[x]):] == 0)
        
        height_diff = np.max(heights) - np.min(heights)
        return holes, height_diff
    
    def play_human(self):
        """人类玩家交互模式"""
        pygame.init()  # 确保pygame初始化
        if self.window is None:
            # 根据游戏区域实际大小设置窗口(10x20方块，每个30像素)
            self.window = pygame.display.set_mode((self.board_width * 30, self.board_height * 30))
            pygame.display.set_caption("Tetris Human Play")
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        self.reset()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.step(0)  # 左移
                    elif event.key == pygame.K_RIGHT:
                        self.step(1)  # 右移
                    elif event.key == pygame.K_UP:
                        # 强制旋转并更新显示
                        old_rotation = self.rotation
                        self.rotation = (self.rotation + 1) % 4
                        if self._check_collision():
                            self.rotation = old_rotation
                        self.render()
                    elif event.key == pygame.K_DOWN:
                        self.step(3)  # 加速下落
                    elif event.key == pygame.K_SPACE:
                        # 直接落到底部
                        while not self._check_collision(0, 1):
                            self.piece_y += 1
                        self._lock_piece()
                        self._clear_lines()
            
            self.render()
            pygame.time.delay(100)  # 控制游戏速度
            
            if self.game_over:
                print(f"Game Over! Score: {self.score}")
                running = False

    def _render_gui(self):
        self.window.fill((0, 0, 0))
        
        # 绘制游戏板
        cell_size = 30
        for y in range(self.board_height):
            for x in range(self.board_width):
                if self.board[y][x]:
                    pygame.draw.rect(self.window, (255, 255, 255), 
                                    (x * cell_size, y * cell_size, cell_size, cell_size))
        
        # 绘制当前方块
        piece_matrix = self._get_current_piece_matrix()
        for y in range(piece_matrix.shape[0]):
            for x in range(piece_matrix.shape[1]):
                if piece_matrix[y][x]:
                    board_x = self.piece_x + x
                    board_y = self.piece_y + y
                    if 0 <= board_y < self.board_height:
                        pygame.draw.rect(self.window, (255, 0, 0), 
                                        (board_x * cell_size, board_y * cell_size, cell_size, cell_size))
        
        # 显示分数和消除行数
        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        lines_text = font.render(f"Lines: {self.lines_cleared}", True, (255, 255, 255))
        self.window.blit(score_text, (self.board_width * cell_size + 20, 20))
        self.window.blit(lines_text, (self.board_width * cell_size + 20, 60))
        
        pygame.display.update()