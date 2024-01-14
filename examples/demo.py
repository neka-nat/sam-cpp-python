import cv2
import numpy as np
import sam_cpp


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # SAMアルゴリズムを使用してセグメンテーションマスクを生成
        masks = predictor.predict(np.array([x, y], dtype=np.float32))
        green_mask = np.zeros_like(image)
        green_mask[:, :, 1] = masks[0]

        # 結果を同じウィンドウで描画
        masked_image = cv2.addWeighted(image, 0.7, green_mask, 0.3, 0)
        cv2.imshow("Image", masked_image)


image = cv2.imread("img.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

predictor = sam_cpp.create_sam_predictor()
predictor.set_image(image)

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", click_event)

while True:
    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

cv2.destroyAllWindows()
