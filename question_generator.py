import random




def select_initial_guess(data):
    """
    Selects an initial guess from the provided data.

    Args:
        data (list): A list of possible guesses. 
        
    Returns:
        str: The selected initial guess.
        
    """
    # For simplicity, we can select the first item in the list as the initial guess.
    # In a more complex implementation, you might want to use some heuristic or random selection.
    if not data:
        raise ValueError("The data list is empty. Cannot select an initial guess.")
    
    return random.choice(data)