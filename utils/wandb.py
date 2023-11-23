import wandb

# wandb log functions
def log_image_report_table(image, pre_report, gt_report, table_name):
    """
    Log a table of image, pre_report, gt_report to wandb
    """
    table = wandb.Table(columns=["Image", "Predicted Report", "Ground Truth Report"])
    for img, pred, targ in zip(image.to("cpu"), pre_report, gt_report):
        print(img.shape)
        img_rgb = img[0].permute(1, 2, 0)
        print(img_rgb.shape)
        table.add_data(wandb.Image(img_rgb.numpy()), pred, targ)
    wandb.log({table_name: table})