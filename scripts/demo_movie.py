from scripts.removal_background import *

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # カメラ画像の横幅を1280に設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # カメラ画像の縦幅を720に設定
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))

    i = 0

    ret, frame = cap.read()

    bg = np.full_like(frame, 255., dtype=float)
    # bg = bg.astype(float)

    t = time() * 1000.

    while True:
        ret, frame = cap.read()

        if not ret:
            continue
        t1 = time() * 1000.
        removed = removal_background(frame, bg)
        t2 = time() * 1000.
        print("ps: ", t2-t1)
        # cv2.imwrite("./examples/outs/{}.png".format(i), removed)

        removed = removed.astype(np.uint8)

        cv2.imshow("removal_background", removed)

        print(time()*1000. - t)
        t = time() * 1000.

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
