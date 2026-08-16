import os
import pygame


class SoundPlayer:
    def __init__(self, sound_dir="assets/sounds"):
        self.sound_dir = sound_dir
        self.cache = {}                  # 音效檔快取:檔名 → pygame Sound 物件
        self.enabled = True
        try:
            pygame.mixer.init()          # 初始化音效系統
        except Exception as e:
            # 音效系統起不來(例如沒音效裝置)就停用,不讓整個程式崩
            print(f"[SoundPlayer] 音效初始化失敗,已停用音效: {e}")
            self.enabled = False

    def play(self, sound_name):
        if not self.enabled or not sound_name:
            return

        # 快取:讀過的音效直接用,沒讀過才載入
        if sound_name not in self.cache:
            path = os.path.join(self.sound_dir, sound_name)
            if not os.path.exists(path):
                self.cache[sound_name] = None      # 檔案不存在,記為 None
            else:
                try:
                    self.cache[sound_name] = pygame.mixer.Sound(path)
                except Exception as e:
                    print(f"[SoundPlayer] 載入音效失敗 {sound_name}: {e}")
                    self.cache[sound_name] = None

        sound = self.cache[sound_name]
        if sound is not None:
            sound.play()                 # 播放(非阻塞,不會卡住主迴圈)