from concurrent.futures import ThreadPoolExecutor
import time

from scripts.removal_background import *

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # カメラ画像の横幅を1280に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # カメラ画像の縦幅を720に設定
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
    cap.set(cv2.CAP_PROP_FPS, 60)

    i = 0

    ret, frame = cap.read()

    bg = np.full_like(frame, 255., dtype=float)
    # bg = bg.astype(float)

    # thread settings
    executor = ThreadPoolExecutor(max_workers=10)
    futures = []

    t1 = time() * 1000.

    while True:
        ret, frame = cap.read()

        if not ret:
            continue
        futures.append(executor.submit(removal_background, frame, bg))

        # cv2.imwrite("./examples/outs/{}.png".format(i), removed)q

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print("running", len(futures))

        for i in reversed(range(len(futures))):
            if not futures[i].running():
                removed = futures[i].result()
                removed = removed.astype(np.uint8)
                print(time() * 1000. - t1)
                t1 = time() * 1000.
                cv2.imshow("removal_background", removed)
                del futures[i]

    executor.shutdown()

    cap.release()
    cv2.destroyAllWindows()
