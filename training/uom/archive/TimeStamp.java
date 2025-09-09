public class TimeStamp {

    //Encapsulation rule
    private int hour;
    private int minute;
    private int second;

    public TimeStamp(int h, int m, int s) {
        hour = h;
        minute = m;
        second = s;
    }

    public void increaseHour() {
        hour++;
        if (hour == 24)
            hour = 0;
    }

    public void printInfo() {
        System.out.println(hour + ":" + minute + ":" + second);
    }
}
