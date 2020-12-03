from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
import bs4

output_file("brushing.html")

x = list(range(-20, 21))
y0 = [abs(xx) for xx in x]
y1 = [xx**2 for xx in x]

# create a column data source for the plots to share
source = ColumnDataSource(data=dict(x=x, x2=x[:-20], y0=y0, y1=y1))

TOOLS = "pan, box_select, lasso_select, box_zoom, help"

# create a new plot and add a renderer
left = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
left.circle('x', 'y0', source=source)

# create another new plot and add a renderer
right = figure(tools=TOOLS, plot_width=300, plot_height=300, title=None)
right.circle('x2', 'y1', source=source)

# left.x_range = right.x_range
p = gridplot([[left, right]])

show(p)

with open('brushing.html') as f:
    soup = bs4.BeautifulSoup(f.read())

print(soup)

style = soup.new_tag('style')
soup.head.append(style)

style.string = '''
.bk-logo-small{
    width: 20px;
    height: 20px;
    background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAIAAAAC64paAAAB1klEQVQ4EZXBIWtyYRgG4PtxQURkTQURDohg2LEIawuCyGzDoOgvMAgLSwdmESaC/gHTklHBZNI2UMF24spARXRNUBB878ELB87H5jd3XUISf7dcLqvVqpDEXxyPR8Mw1uu1UkpI4mKlUsnn872+vlqW1Wg0hCQu0Gw2LctKpVLz+RwASQBCEr+5u7ubzWatVuvx8RFAPp/v9XoAhCTOIxkOhzebjYiQhEYSmpDEGYfDIRAInE6n1WqVTCY/Pz8B3N/fD4dDaEISPzkej16vF0AkEjEM4+3tDdrpdPJ4PNCEJL5ZLBbRaBQ/IQmHkMQ3IgJtMpkUi8WPjw9ooVBovV7DISTxr6urK6UUAJ/Pt9/vRQSOp6endrsNh5CESyaTGY1G0JRS19fXu90Ojul0ent7C4eQhOPl5eX5+RkOkiICF5JwEZLQttttMBiEJiKFQqHRaMRiMbgopUQEDiEJLR6Pv7+/+/3+fr+fzWahHQ4HwzA2mw00knARkgAeHh4Gg0G9Xq/Vavim0+lUKhUAJOEiJAGIyHg8TqfTOKPZbFqWpZQSETiEZCKR6Ha7qVQK/5XL5YbDIVxEKWXbtmma+A1J27ZN04RDyuVyt9vFZWzbvrm5geMLMe7wzJiEXNcAAAAASUVORK5CYII=) !important;}
'''

#save our edits
with open("brushing.html", "w") as f:
    f.write(str(soup))
