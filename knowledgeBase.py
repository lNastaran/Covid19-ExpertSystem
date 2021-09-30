def engine(fever, headAche, pain, weakness, runnyNose, sneezing, soreThroat, cough, asthma, gender, age, anosmia, digestive):
 if pain <= 0.5:
  if cough <= 0.5:
   if sneezing <= 0.5:
    if runnyNose <= 0.5:
     return 5
    else:

     return 1
   else:

    if anosmia <= 0.5:
     if runnyNose <= 0.5:
      if gender <= 0.5:
       return 3
      else:

       return 5
     else:

      return 3
    else:

     return 1
  else:

   if soreThroat <= 0.5:
    if runnyNose <= 0.5:
     return 3
    else:

     return 2
   else:

    if digestive <= 0.5:
     if fever <= 0.5:
      return 1
     else:

      if age <= 0.5:
       return 1
      else:

       return 4
    else:

     if anosmia <= 0.5:
      return 2
     else:

      return 4
 else:

  if asthma <= 0.5:
   return 2
  else:

   if age <= 0.5:
    return 2
   else:

    return 4
