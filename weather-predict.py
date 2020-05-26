import sys
import getopt
import tensorflow as tf
import locale
import numpy as np
import pandas as pd
import banner
from datetime import datetime
import matplotlib.pyplot as plt
from astral import LocationInfo
from astral.sun import sun
from tensorflow.python.util import deprecation
from datetime import timedelta
deprecation._PRINT_DEPRECATION_WARNINGS = False


def date_info(dt):
    city = LocationInfo("Moscow", "Russia", "Europe/Moscow", 55.751244, 37.618423)
    s = sun(city.observer, date=dt)
    def sec(x): return (x - x.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return [sec(s["dawn"]), sec(s["sunrise"]), sec(s["noon"]), sec(s["sunset"]), sec(s["dusk"])]


def datetime_to_array(d):
    return [d.day, d.month, d.year, date_to_nth_day(d)]+date_info(d)


def date_to_array(date):
    return datetime_to_array(datetime.strptime(date, '%Y-%M-%d'))


def date_to_nth_day(date):
    # date = datetime.datetime.strptime(date, format=format)
    new_year_day = datetime(year=date.year, month=1, day=1)
    return (date - new_year_day).days + 1


def traindata():
    locale.setlocale(locale.LC_ALL, "ru_RU")

    def dateparse(x):
        return datetime.strptime(x, '%d %B %Y')

    def day(day):
        return day.dayofyear

    df = pd.read_csv("msk_tmp.csv", sep=";", parse_dates=[0])  # , date_parser=dateparse)
    if 'sunrise' in df:
        return df
    else:
        df["dayofyear"] = df['date'].apply(lambda x: x.dayofyear).astype('int')
        df["day"] = df['date'].apply(lambda x: x.day).astype('int')
        df["month"] = df['date'].apply(lambda x: x.month).astype('int')
        df["year"] = df['date'].apply(lambda x: x.year).astype('int')

        df["dawn"] = df['date'].apply(lambda x: date_info(x)[0]).astype('float64')
        df["sunrise"] = df['date'].apply(lambda x:  date_info(x)[1]).astype('float64')
        df["noon"] = df['date'].apply(lambda x:  date_info(x)[2]).astype('float64')
        df["sunset"] = df['date'].apply(lambda x: date_info(x)[3]).astype('float64')
        df["dusk"] = df['date'].apply(lambda x: date_info(x)[4]).astype('float64')

        df.to_csv('msk_tmp.csv', sep=";")

        return df


def nn_model(input_count):
    model = tf.keras.Sequential()

    # elu 100i loss:4 lr:0.000003
    # elu 200i loss:3.8
    # elu 300i loss:3.8
    # elu 400i loss:3.79
    # elu 500i loss:3.79 lr:0.003
    # elu 600i loss:3.79 finish
    # relu 600i loos:9.1 lr:0.003
    # softmax 600i loos:9.1 lr:0.003
    # with date info
    # elu 100i loss:9.1 lr:0.000003 -
    # elu 200i loss:9.1 -
    # elu 300i loss:9.1666 lr:0.003 =
    # elu 400i loss:9.1666 lr:0.0003 =
    # elu 500i loss:9.1666 lr:0.03 =
    # relu 600i loss:9.2 lr:0.03 -
    # relu 700i loss:9.1 lr:0.03 -
    # relu 1000i loss:9.1 lr:0.03 =
    # softmax 1300i loss:9.1 lr:0.03 =
    # sigmoid 1600i loss:6.78 lr:0.03 -
    # sigmoid 1900i loss:4.9339 lr:0.03 =
    # hard_sigmoid 1900i loss:4.9339 lr:0.03 =
    # tanh 1900i loss:5.4 lr:0.03
    # exponential 1900i loss:nan lr:0.03

    # act = 'sigmoid' # 4.8
    # model.add(tf.keras.layers.Dense(32, input_shape=(xdata.shape[1],), activation=act, kernel_initializer='he_uniform'))
    # model.add(tf.keras.layers.Dense(16, activation=act))
    # model.add(tf.keras.layers.Dense(8, activation='elu'))
    # model.add(tf.keras.layers.Dense(8, activation=act))
    # model.add(tf.keras.layers.Dense(4, activation='elu'))
    # model.add(tf.keras.layers.Dense(72))
    # model.add(tf.keras.layers.Dense(1))

    # act = 'sigmoid' # 3.70
    # model.add(tf.keras.layers.Dense(32*10, input_shape=(xdata.shape[1],), activation=act))  # , kernel_initializer='he_uniform'))
    # model.add(tf.keras.layers.Dense(1))

    # 9-18

    act = 'sigmoid'  # 0.3324 0.2100
    model.add(tf.keras.layers.Dense(32*20, input_shape=(input_count,), activation=act))  # , kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(32*20, activation=act))
    model.add(tf.keras.layers.Dense(32*10, activation=act))
    model.add(tf.keras.layers.Dense(1))

    # opt = tf.keras.optimizers.RMSprop(learning_rate=0.000003, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False)
    opt = tf.keras.optimizers.Adam(learning_rate=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

    # model.compile(loss='mean_absolute_error', optimizer=opt)  # optimizer: tf.train.sgd(0.001)});
    model.compile(loss='mean_squared_logarithmic_error', optimizer=opt)  # optimizer: tf.train.sgd(0.001)});
    model.summary()

    try:
        model.load_weights("weather.model/weather.weights")
    except:
        print("new model")
    return model


def nn_train(model,xdata, ydata, pdata,epochs=10):
    # xs = tf.constant(xdata, dtype='float32')
    # ys = tf.constant(ydata, dtype='float32')

    hist = model.fit(xdata, ydata, epochs=epochs,  verbose=1,validation_split=0.9)
    model.save_weights("weather.model/weather.weights")

    result = nn_predict(model,pdata)

    plt.subplot(311)
    plt.plot(result.round(1))
    plt.title("Result")

    plt.subplot(312)
    plt.plot(hist.history["loss"], color='green')
    plt.title("Loss")

    plt.subplot(313)
    plt.plot(ydata, color='#aabbcc')
    plt.title("Train data")

    plt.show()


def nn_predict(model,pdata):
    return model.predict(pdata)


try:
    opts, args = getopt.getopt(sys.argv[1:], "tdb", ["train", "demo", "epochs=", "banner"])
except getopt.GetoptError:
    print('--train --demo')
    sys.exit(2)


train = False

if len(opts) > 0:
    for o, a in opts:
        if o in ("-d", "--demo"):
            print("demo")
            model = nn_model(9)

            tmp = []
            for i in range(5):
                tmp.append(datetime_to_array(datetime.now() + timedelta(days=i)))
            print(tmp)
            print(nn_predict(model, np.array(tmp)).round(1))
        elif o in ("-b", "--banner"):
            model = nn_model(9)
            tmp = []
            for i in range(365):
                tmp.append(datetime_to_array(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0, month=1, day=1) + timedelta(days=i)))
            result = nn_predict(model, np.array(tmp)).round(1)

            plt.plot(result)
            #plt.xlabel('Day of 2020 year')
            #plt.ylabel('C°')
            # plt.show()

            banner.banner(10, 10, plt)
        elif o in ("-t", "--train"):
            train = True
        elif o in ("--epochs"):
            epochs = a
else:
    print("--demo --train --banner")

if train is True:
    df = traindata()

    print(df)

    tmp = []
    for i in range(365):
        tmp.append(datetime_to_array(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0, month=1, day=1) + timedelta(days=i)))

    model = nn_model(9)
    nn_train(model,df[['year', 'month', 'day', 'dayofyear', 'dawn', 'sunrise', 'noon', 'sunset', 'dusk']].to_numpy(), df.avg.to_numpy(), np.array(tmp),int(epochs))
