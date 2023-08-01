import click

from driver.driver import MMMDriver


@click.command()
@click.option("-f", "--input_filename", required=True, help="input data filename")
def main(input_filename: str):
    driver = MMMDriver()
    driver.main(input_filename)


if __name__ == "__main__":
    main()
