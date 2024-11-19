from django.shortcuts import render
from django.http import JsonResponse
from .violet import get_response  # Import the function from the new script

def index(request):
    return render(request, 'ai_interface/index.html')

def prompt_interface(request):
    if request.method == 'POST':
        user_input = request.POST.get('prompt')
        # Call the get_response function with the user input
        response = get_response(user_input)
        return JsonResponse({'response': response})
    return render(request, 'ai_interface/prompt_interface.html')