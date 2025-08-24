buildings = {
    "gfgsf fgfsg  fsgsf  fg  fgs": "UAE",
    "dgfsd fgsf fs gsf fgfgte ": "USA",
    "eyydbcvsdfkv v bfdkv bfv efv kv bwfiv vbf ": "UK",
    "ruyuy vdcj vb ruofgyw": "Italia",
    "gfgsf fsf  fg  fgs": "France",
    "dgfsd fgsffgte ": "Netherlands",
    "eyydbbfdkv bfv efv kv bwfiv vbf ": "Greece",
    "ru vb ruofgyw": "Imalia",
    
}

country_build = {
    "Imalia": "AAAAAdsf fgre vfve",
    "Italia": "BBBBB ruofgyw",
    "France": "FFFFFFF fg  fgs"
}


for key, value in buildings.items():
    if value not in country_build:
        country_build[value] = key
    else:
        country_build[value] = f"{country_build[value]} {key}"

for key in country_build.keys():
    print(f"{key} : {country_build[key]}")