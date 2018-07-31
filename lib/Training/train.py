import time
import torch
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy

def train(dataset, gen_model, disc_model, criterion, epoch, gen_optimizer, disc_optimizer, lr_scheduler, device, args):
    gen_losses = AverageMeter()
    disc_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    gen_model.train()
    disc_model.train()

    end = time.time()

    for i, (input, target) in enumerate(dataset.train_loader):
        input, target = input.to(device), input.to(device)
        data_time.update(time.time() - end)

        lr_scheduler.adjust_learning_rate(gen_optimizer, i + 1)
        lr_scheduler.adjust_learning_rate(disc_optimizer, i + 1)

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

        disc_optimizer.zero_grad()  
        disc_loss.backward()
        disc_optimizer.step()

        gen_out = gen_model(z)
        disc_fake = disc_model(gen_out)
        gen_loss = criterion(disc_fake, y_real)

        gen_losses.update(gen_loss.item(), input.size(0))

        gen_optimizer.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()
        del input, target, z, y_real, y_fake, disc_real, gen_out, disc_fake

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Generator Loss {gen_losses.val:.4f} ({gen_losses.avg:.4f})\t'
                  'Discriminator Loss {disc_losses.val:.3f} ({disc_losses.avg:.3f})\t'.format(
                   epoch, i, len(dataset.train_loader), batch_time=batch_time,
                   data_time=data_time, gen_losses=gen_losses, disc_losses=disc_losses))
        
    print(' * Train: Generator Loss {gen_losses.avg:.3f} Discriminator Loss {disc_losses.avg:.3f}'\
        .format(gen_losses=gen_losses, disc_losses=disc_losses))
    print('-' * 80)
    return disc_losses.avg, gen_losses.avg




