import helper as h
import argparse
import torch


# command line arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('checkpoint')
parser.add_argument('--top_k', default=1, dest='top_k')
parser.add_argument('--category_names', default='cat_to_name.json', dest='category_names')
parser.add_argument('--gpu', action='store_true', default=False, dest='gpu')
args = parser.parse_args()


# prediction
def predict(image_path, model, topk):

    model.to(device)
    model.eval()

    imaget = process_image(image_path)
    imaget.to(device)

    imaget = imaget.unsqueeze(0)
    image = imaget.type(torch.cuda.FloatTensor)
    output = model.forward(image)
    ps = torch.exp(output)
    model.train()

    return ps.topk(topk)


# sanity checking function
def sanity_check(image_path):

    probs, labels = predict(image_path, model, args.top_k)

    ps = [x for x in probs.cpu().detach().numpy()[0]]
    npar = [x for x in labels.cpu().numpy()[0]]
    names = list()

    inv_mapping = {v: k for k, v in model.class_to_idx.items()}

    for i in npar:
        names.append(cat_to_name[str(inv_mapping[i])])


    imshow(process_image(image_path), ax=plt.subplot(2,1,1));
    plt.title(cat_to_name['21'])

    plt.subplot(2,1,2)
    sb.barplot(y=names, x=ps, color=sb.color_palette()[0]);
    plt.show()

# predicting flower name
cat_to_name = h.label_mapping('cat_to_name.json')

# loading model from checkpoint
model, optimizer = h.load_checkpoint(args.checkpoint)
device = "cuda:0" if args.gpu else "cpu"

# predicting image
sanity_check(args.input)
