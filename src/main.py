from controller import Controller

from weap_util.weap_container import run

def main():
    controller = Controller()
    run(controller, "../assets/config.yaml")

if __name__ == "__main__":
    main()