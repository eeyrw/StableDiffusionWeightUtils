from jinja2 import Environment, FileSystemLoader
import os



def genReport(title,reportItems):
    root = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(root, 'templates')
    env = Environment( loader = FileSystemLoader(templates_dir) )
    template = env.get_template('DataComparison.jinja')


    filename = os.path.join(root, 'DataComparison.html')
    with open(filename, 'w') as fh:
        fh.write(template.render(
            report_title = title,
            items=reportItems
        ))