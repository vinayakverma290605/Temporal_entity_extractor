import numpy as np # Add this import at the top
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .nlp_engine import get_temporal_entities

def home(request):
    return render(request, 'index.html')

@csrf_exempt
def analyze(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        entities = get_temporal_entities(text)
        
        data = []
        for e in entities:
            # This line converts the 'np.float32' to a normal number
            score_fixed = float(e['score'])
            
            data.append({
                'word': e['word'],
                'type': e['entity_group'],
                'score': score_fixed
            })
        return JsonResponse({'entities': data})