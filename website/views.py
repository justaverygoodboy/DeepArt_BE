import json,os,cv2,base64,time
import matplotlib.pyplot as plt
import tensorflow as tf
from website.utils.photo2Cartoon import Photo2Cartoon
from website.utils.colorizers import siggraph17
from website.utils.colorizers.util import *
from website.utils import L0smooth, RRDBNet_arch as arch
from django.http import HttpResponse
from imageio import imread, imsave
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(BASE_DIR,"website/save_path")

def base64_cv2_url(imgStr):
    img = imgStr.split('base64,')
    imgData = base64.b64decode(img[1])
    timestamp = str(int(time.time()))
    file_url = os.path.join(BASE_DIR,'website/images/%s.%s' % (timestamp, 'jpg'))

    downImg = open(file_url,'wb')
    downImg.write(imgData)
    downImg.close()
    return img[0],file_url

def cv2_base64(head,res_url):
    bb=base64.b64encode(open(res_url, 'rb').read())
    lens = len(bb)
    resStr = head+'base64,'+str(bb)[2:lens]
    return resStr

def get_body(request):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    body = request.body.decode('utf-8')
    return json.loads(body)

#AI彩色化
def colorization(request):
    body = get_body(request)
    imgStr = body['img']
    head, img_url = base64_cv2_url(imgStr)
    colorizer_siggraph17 = siggraph17.siggraph17(pretrained=True).eval()
    if torch.cuda.is_available():
        colorizer_siggraph17.cuda()
    img = np.asarray(cv2.imread(img_url,cv2.COLOR_RGBA2RGB))
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
    if torch.cuda.is_available():
        tens_l_rs = tens_l_rs.cuda()
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    img_name = "result1.png"
    res_url = os.path.join(save_path, img_name)
    plt.imsave(res_url, out_img_siggraph17)
    resStr = cv2_base64(head,res_url)
    return HttpResponse(resStr)

#头像卡通化
def cartoon(request):
    body = get_body(request)
    imgStr = body['img']
    head, img_url = base64_cv2_url(imgStr)
    img = cv2.imread(img_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb= cv2.cvtColor(Photo2Cartoon().inference(img), cv2.COLOR_BGR2RGB)
    img_name = "result2.png"
    res_url = os.path.join(save_path,img_name)
    cv2.imwrite(res_url, img_rgb)
    resStr = cv2_base64(head,res_url)
    return HttpResponse(resStr)


#超分辨率重建
def sr(request):
    body = get_body(request)
    imgStr = body['img']
    head, img_url = base64_cv2_url(imgStr)
    model_path = os.path.join(BASE_DIR,"website/pretrained_models/RRDB_ESRGAN_x4.pth")
    torch.cuda._initialized = True
    path = img_url
    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval()
    device = torch.device("cpu")
    model = model.to(device=torch.device("cpu"))
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    img_name = "result3.png"
    res_url = os.path.join(save_path, img_name)
    cv2.imwrite(res_url, output)
    resStr = cv2_base64(head,res_url)
    return HttpResponse(resStr)

#图像风格迁移
def style(request):
    body = get_body(request)
    imgStr = body['img']
    head, img_url = base64_cv2_url(imgStr)
    style = body['type']
    model = os.path.join(BASE_DIR,"website/pretrained_models/style_model",'samples_%s' % style)
    img_name = "4.jpg"
    res_url = os.path.join(save_path, img_name)
    X_image = imread(img_url)
    sess = tf.compat.v1.Session()
    tf.compat.v1.disable_eager_execution()
    sess.run(tf.compat.v1.global_variables_initializer())
    tf.compat.v1.disable_v2_behavior()
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(model, 'fast_style_transfer.meta'))
    saver.restore(sess, tf.compat.v1.train.latest_checkpoint(model))
    graph = tf.compat.v1.get_default_graph()
    X = graph.get_tensor_by_name('X:0')
    g = graph.get_tensor_by_name('transformer/g:0')
    gen_img = sess.run(g, feed_dict={X: [X_image]})[0]
    gen_img = np.clip(gen_img, 0, 255) / 255.
    imsave(res_url, gen_img)
    resStr = cv2_base64(head, res_url)
    return HttpResponse(resStr)

#图像增强
def L0Smoothing(request):
    body = get_body(request)
    imgStr = body['img']
    head, img_url = base64_cv2_url(imgStr)
    load_size = 360
    win_size = 600
    win_size = int(win_size / 4.0) * 4
    ls = L0smooth.L0smoothing(load_size=load_size, win_size=win_size)
    ls.read_image(img_url)
    ls.L0Smoothing()
    img_name = "result5.png"
    res_url = os.path.join(save_path,img_name)
    ls.save_result(res_url)
    resStr = cv2_base64(head,res_url)
    return HttpResponse(resStr)


