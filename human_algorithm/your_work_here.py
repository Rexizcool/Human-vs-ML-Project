def human_classify2():
    if culmen_length_mm >= 45 and island == 'Dream':
        species = 'Chinstap'
        elif culmen_length_mm >= 45 and island == 'Biscoe':
        species = 'Gentoo'
    elif culmen_length_mm < 45:
        species = 'Adelie'