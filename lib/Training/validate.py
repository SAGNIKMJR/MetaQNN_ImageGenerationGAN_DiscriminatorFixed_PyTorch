import time
import torch
import os
from torchvision.utils import save_image
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy

def validate(dataset, gen_model, disc_model, criterion, epoch, device, args, save_path_pictures):
    batch_time = AverageMeter()
    losses = AverageMeter()
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()

    gen_model.eval()
    disc_model.eval()

    end = time.time()

    for i, (input, target) in enumerate(dataset.val_loader):
        with torch.no_grad():
            input, target = input.to(device), input.to(device)

            if args.no_gpus>1:
                input_size = gen_model.module.input_size
            else:
                input_size = gen_model.input_size

            # set inputs and targets
            z = torch.randn((input.size(0), input_size)).to(device)
            y_real, y_fake = torch.ones(input.size(0), 1).to(device), torch.zeros(input.size(0), 1).to(device)

            disc_real = disc_model(input)
            gen_out = gen_model(z)
            disc_fake = disc_model(gen_out)

            disc_real_loss = criterion(disc_real, y_real)
            disc_fake_loss = criterion(disc_fake, y_fake)
            disc_loss = disc_real_loss + disc_fake_loss

            disc_losses.update(disc_loss.item(), input.size(0))

            gen_out = gen_model(z)
            disc_fake = disc_model(gen_out)
            gen_loss = criterion(disc_fake, y_real)

            gen_losses.update(gen_loss.item(), input.size(0))

            if i % args.print_freq == 0:
                save_image((gen_out.data.view(-1,input.size(1),input.size(2),input.size(3))), os.path.join(save_path_pictures, 'sample_epoch_' + str(epoch) + '_ite_'+str(i+1)+'.png'))
            del input, target, z, y_real, y_fake, disc_real, gen_out, disc_fake

    print(' * Validate: Generator Loss {gen_losses.avg:.3f} Discriminator Loss {disc_losses.avg:.3f}'\
        .format(gen_losses=gen_losses, disc_losses=disc_losses))
    print('-' * 80)

    return disc_losses.avg, gen_losses.avg