### demo master file

import sys
from workflows.automation import workFlows
from apps.calendar import calendarAgent


def main():
    while True:
        print("google agentic suite implementation demonstration\n")
        print("1. Gmail → Calendar (create events from unread emails)")
        print("2. Gmail → Drive (download attachments → upload to Drive)")
        print("3. Show upcoming Calendar events")
        print("0. Exit")

        choice = input("Select an option: ").strip()

        if choice == "1":
            workFlows.gmail_to_calendar()
        elif choice == "2":
            workFlows.gmail_to_drive()
        elif choice == "3":
            calendarAgent.list_events(max_results=5)
        elif choice == "0":
            print("exit")
            sys.exit(0)


if __name__ == "__main__":
    main()

