첨엔..... 아 이게 loss가 지금 3개라서... 두 개의 변수로 total loss 구해야해서...
total_loss = loss_cont + (b x (a x loss_kd_genre + (1-a) x loss_kd_content) 막 이러니까 넘 복잡한거야

그래가지고 배울 수 있는 loss 만들었더니 모델이 빠져가지고 그냥 loss 줄일려고 kd loss들 싸그리 빼고 loss_cont만 남기고
막 그래서 train_loss 는 막 마이너스 찍고 mrr도 떨어지고 이거 원..