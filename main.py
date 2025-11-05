from lib import *
from dataset import *
from imagetransform import *
from train_model import *

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def main():
    # ==== 1️Cấu hình dữ liệu ====
    train_dir = 'Dataset_split/train'
    val_dir = 'Dataset_split/val'

    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(resize, mean, std)

    train_dataset = SatelliteDataset(train_dir, transform=transform.data_transform['train'])
    val_dataset = SatelliteDataset(val_dir, transform=transform.data_transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    dataloader_dict = {'train': train_loader, 'val': val_loader}

    # ==== 2️Chuẩn bị mô hình ====
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Nếu muốn chỉ fine-tune classifier cuối:
    # for param in model.features.parameters():
    #     param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, 3)  # 3 lớp: rain, medium_rain, heavy_rain
    model = model.to(device)

    # ==== Cấu hình huấn luyện ====
    criterion = nn.CrossEntropyLoss()
    params1, params2, params3 = update_params(model)
    print(len(params1), len(params2), len(params3))
    optimizer = optim.Adam([
        {'params': params1, 'lr': 1e-4},
        {'params': params2, 'lr': 1e-4},
        {'params': params3, 'lr': 5e-4}
    ])

    # ==== Huấn luyện ====
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    num_epochs = 30
    train_losses, val_losses, train_accs, val_accs = train_model(model, dataloader_dict, criterion, optimizer, num_epochs, scheduler)

    # ====  Lưu mô hình ====
    save_dir = r"D:\Documents\BaiTap\AI\DoAnAI\Satelitte"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model, 'satelite_model.pth')

    # ==== Vẽ biểu đồ ====
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss per Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy per Epoch')

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()