#生成验证码
import random,string
from io import BytesIO
from PIL import Image,ImageFont,ImageDraw

class ImageCode:
    def rand_color(self):
        red = random.randint(32,127)
        green = random.randint(32,127)
        blue = random.randint(32,127)
        return red,green,blue
    def gen_text(self):
        # sample用于从一个大的列表或字符串中，随机取得N个字符串，来构建出一个子列表
        list = random.sample(string.ascii_letters+string.digits,5)
        return (''.join(list))

    #绘制干扰线
    def draw_lines(self,draw,num,width,height):
        for num in range(num):
            x1 = random.randint(0,width / 2)
            y1 = random.randint(0, height / 2)
            x2 = random.randint(0, width)
            y2 = random.randint(height/2, height)
            draw.line(((x1,y1),(x2,y2)),fill='black',width=2)

    #绘制验证码图片
    def draw_verify_code(self):
        code = self.gen_text()
        width,height = 120,50
        im = Image.new('RGB',(width,height),'white')
        font = ImageFont.truetype(font='arial.ttf',size=40)
        draw = ImageDraw.Draw(im)
        for i in range(5):
            draw.text((5+random.randint(-3,3)+23*i,5+random.randint(-3,3)),
                      text=code[i],fill=self.rand_color(),font=font)
        self.draw_lines(draw,4,width,height)
        # im.show()
        return im,code

    #生成图片验证码并返回给控制器
    def get_code(self):
        image,code = self.draw_verify_code()
        buf = BytesIO()
        image.save(buf,'png')
        bstring = buf.getvalue()
        return code,bstring

# ImageCode().gen_text( )
# ImageCode().draw_verify_code()