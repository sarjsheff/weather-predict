from wand.image import Image
from wand.drawing import Drawing
from io import BytesIO


def banner(temp, loss,resultplt=None):
    with Image(width=140, height=230, background="white") as img:
        draw = Drawing()
        draw.font = 'font/OpenSans-Bold.ttf'
        draw.font_size = 14
        draw.font_antialias = True
        draw.text_alignment = 'center'
        draw.text(70, 14, 'Moscow')
        draw(img)

        draw = Drawing()
        draw.font = 'font/OpenSans-SemiBold.ttf'
        draw.font_size = 50
        draw.font_antialias = True
        draw.text_alignment = 'center'
        draw.text(70, 70, "%+d" % float(temp)+'°')
        draw(img)
 
        if resultplt:
            image_data = BytesIO()
            resultplt.axis('off')
            resultplt.gcf().set_size_inches(1.4, 0.7)
            resultplt.gcf().set_dpi(100)
            resultplt.tight_layout()
            resultplt.savefig(image_data, format='png')
            image_data.seek(0)
            result_image = Image(file=image_data)
            img.composite(image=result_image,left=0,top=110)

            draw = Drawing()
            draw.font = 'font/OpenSans-Bold.ttf'
            draw.font_size = 14
            draw.font_antialias = True
            draw.text_alignment = 'center'
            draw.text(70, 115, '2020')
            draw(img)


        for i, t in enumerate(["loss: "+str(loss), "input: 9 params", "train data: 16430 rows"]):
            draw = Drawing()
            draw.font = 'font/OpenSans-Light.ttf'
            draw.font_size = 10
            draw.font_antialias = True
            draw.text(4, 190+(i*12), t)
            draw(img)

        img.save(filename='weather.png')


# banner("-29.1", "0.1414")
