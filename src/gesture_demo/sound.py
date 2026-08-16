import os
import pygame


class SoundPlayer:
    def __init__(self, sound_dir="assets/sounds"):
        self.sound_dir = sound_dir
        self.cache = {}
        self.enabled = True
        self.channel = None                    # 固定用這個 channel 播
        try:
            pygame.mixer.init()
            self.channel = pygame.mixer.Channel(0)   # 拿第 0 號 channel
        except Exception as e:
            print(f"[SoundPlayer] 音效初始化失敗,已停用: {e}")
            self.enabled = False

    def play(self, sound_name):
        if not self.enabled or not sound_name:
            return

        if sound_name not in self.cache:
            path = os.path.join(self.sound_dir, sound_name)
            if not os.path.exists(path):
                self.cache[sound_name] = None
            else:
                try:
                    self.cache[sound_name] = pygame.mixer.Sound(path)
                except Exception as e:
                    print(f"[SoundPlayer] 載入失敗 {sound_name}: {e}")
                    self.cache[sound_name] = None

        sound = self.cache[sound_name]
        if sound is not None:
            self.channel.play(sound)           # 在固定 channel 播,自動蓋掉舊的

    def stop(self):
        if self.enabled and self.channel is not None:
            self.channel.stop()      # 停掉這個 channel 正在播的聲音