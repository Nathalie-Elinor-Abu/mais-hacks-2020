from django.shortcuts import render


# our home page view
def home(request):
    return render(request, 'index.html')


# custom method for generating predictions
def get_predictions(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    import pickle
    model = pickle.load(open("../../model/model.sav", "rb"))  # todo use our ml model
    prediction = model.predict(model.transform())

    if prediction == 0:
        return "not survived"
    elif prediction == 1:
        return "survived"
    else:
        return "error"


# our result page view
def result(request):
    r = 'hello world'
    return render(request, 'result.html', {'result': r})
