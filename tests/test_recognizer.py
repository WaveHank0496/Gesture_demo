from src.gesture_demo.recognizer import distance


def test_distance():
    # 3-4-5 直角三角形,距離應該是 5
    assert distance((0, 0, 0), (3, 4, 0)) == 5.0
#    assert distance((0, 0, 0), (3, 4, 0)) == 99.0

    # 同一點,距離 0
    assert distance((1, 1, 0), (1, 1, 0)) == 0.0
    print("test_distance 通過")


if __name__ == "__main__":
    test_distance()
    print("全部測試通過")