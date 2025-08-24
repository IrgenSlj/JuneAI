family = {
    "father": {
        "name" : "John",
        "born" : 1960,
        "quotes" : {
            "funny" : "Tssuuuppp!!",
            "serious": "Listen to me",
            "sad":"I feel blue"
        }
    },

    "mother": {
        "name" : "Marina",
        "born" : 1964,
        "quotes" : {
            "funny" : "Hiiiihhihihi!!",
            "serious": "Not me!",
            "sad":"I have a headache"
        }
    },

    "children": [
        {
            "name": "Emma",
            "age": 12,
            "hobby": "tennis"
        },
        {
            "name": "Mike",
            "age": 14,
            "hobby": "football"
        },
        {
            "name": "Saimir",
            "age": 17,
            "hobby": "nothing"
        }
    ]
}

for i in range(len(family["children"])):
    x = f'{family["children"][i]["name"]}, {family["children"][i]["age"]}, {family["children"][i]["hobby"]}'
    print(x)
print()

