package LDA;

public class TestLDA_ES {
	public static void main(String[] args){
		long elapsedTime = System.currentTimeMillis();
		if (args.length != 1) {
		    System.out.println("Usage:java -cp *.jar LDA.TestLDA_ES topicnum");
		    System.exit(1);
        }
		LDA_ES lda = new LDA_ES(Integer.valueOf(args[0]));
//		LDA_ES lda = new LDA_ES(10);
		lda.solve();
		elapsedTime = (System.currentTimeMillis() - elapsedTime)/1000;
		System.out.println("程序耗时：" + elapsedTime/3600 + ":" + elapsedTime/60%60 + ":" + elapsedTime%60);
		System.out.println("finished!");
	}
}
