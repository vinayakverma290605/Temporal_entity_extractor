TEST_CASES = [
    # English
    {
        "text": "The meeting is on 29 June at 10 AM.",
        "expected": ["29 June", "10 AM"]
    },
    {
        "text": "She left 3 days ago and will return next Friday.",
        "expected": ["3 days ago", "next Friday"]
    },
    {
        "text": "The event was held in August 2023.",
        "expected": ["August 2023"]
    },
    {
        "text": "Call me tomorrow morning.",
        "expected": ["tomorrow", "morning"]
    },
    {
        "text": "The deadline is Q3 2024.",
        "expected": ["Q3 2024"]
    },

    # Hindi
    {
        "text": "कल सुबह बैठक है।",
        "expected": ["कल", "सुबह"]
    },
    {
        "text": "अगले हफ्ते परीक्षा है।",
        "expected": ["अगले हफ्ते"]
    },
    {
        "text": "विनायक सोमवार को आएगा।",
        "expected": ["सोमवार"]
    },
    {
        "text": "पिछले महीने दिल्ली में बाढ़ आई।",
        "expected": ["पिछले महीने"]
    },
    {
        "text": "3 दिन पहले वो चला गया।",
        "expected": ["3 दिन पहले"]
    },

    # Bengali
    {
        "text": "আগামীকাল পরীক্ষা আছে।",
        "expected": ["আগামীকাল"]
    },
    {
        "text": "গত সপ্তাহে আমরা ঢাকা গিয়েছিলাম।",
        "expected": ["গত সপ্তাহ"]
    },
    {
        "text": "সোমবার সকালে মিটিং আছে।",
        "expected": ["সোমবার", "সকাল"]
    },
    {
        "text": "আগামী মাসে পরীক্ষা শুরু হবে।",
        "expected": ["আগামী মাসে"]
    },
    {
        "text": "কিছুদিন ধরে বৃষ্টি হচ্ছে।",
        "expected": ["কিছুদিন ধরে"]
    },
]