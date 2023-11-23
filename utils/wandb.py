import wandb

# wandb log functions
def image_report_table(image, pre_report, gt_report):
    """
    Log a table of image, pre_report, gt_report to wandb
    """
    table = wandb.Table(columns=["Image", "Predicted Report", "Ground Truth Report"])
    for img, pred, targ in zip(image.to("cpu"), pre_report, gt_report):
        img_rgb = img[0].permute(1, 2, 0)
        table.add_data(wandb.Image(img_rgb.numpy()), pred, targ)
    return table