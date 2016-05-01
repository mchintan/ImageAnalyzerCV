/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package imageanalyzercv;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.highgui.Highgui;



/**
 *
 * @author chintan
 */
public class ImageAnalyzerCV {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        System.out.println("path: "+System.getProperty("java.library.path"));
        System.loadLibrary("opencv_java300");
         
        Mat m = Highgui.imread("/Users/chintan/Downloads/software/image_analyis/mydata/SAM_0763.JPG");
        System.out.println("m = " + m.height());
        MatOfKeyPoint points =new MatOfKeyPoint();
        FeatureDetector.create(FeatureDetector.SURF).detect(m, points);
        
       
        
        Mat m2 = Highgui.imread("/Users/chintan/Downloads/software/image_analyis/mydata/SAM_0764.JPG");
        System.out.println("m = " + m2.height());
        MatOfKeyPoint points2 =new MatOfKeyPoint();
        FeatureDetector.create(FeatureDetector.SURF).detect(m2, points2);
        
        DescriptorExtractor SurfExtractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);
        Mat imag1Desc = new Mat();
        SurfExtractor.compute(m, points, imag1Desc);
        
        Mat imag2Desc = new Mat();
        SurfExtractor.compute(m2, points2, imag2Desc);
        
        MatOfDMatch matches = new MatOfDMatch();
        
        Mat imgd = new Mat();
        imag1Desc.copyTo(imgd);
        System.out.println(imgd.size());
        DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING).match(imag2Desc, imag1Desc, (MatOfDMatch) matches);
        
        double min_distance=1000.0;
        double max_distance=0.0;
        DMatch[] matchArr = matches.toArray();
        for(int i=0;i<matchArr.length;i++)
        {
            if(matchArr[i].distance > max_distance) max_distance = matchArr[i].distance;
            if(matchArr[i].distance < min_distance) min_distance = matchArr[i].distance;
        }
        
        ArrayList<DMatch> good_matches = new ArrayList<DMatch>();
        
        System.out.println("Min Distance: "+min_distance+"  Max distance: "+max_distance);
        double totalScore=0.0;
        for(int j=0;j<imag1Desc.rows() && j<matchArr.length ;j++)
        {
            if ((matchArr[j].distance <= (11 * min_distance)) && (matchArr[j].distance >= min_distance*1)) {
                good_matches.add(matchArr[j]);
                //System.out.println(matchArr[j]);
                totalScore=totalScore+matchArr[j].distance;

            }
            //good_matches.add(matchArr[j]);
           
        }
        System.out.println((1-(totalScore/(good_matches.size()*((max_distance+min_distance)/2))))*100);
       // System.out.println(matches.toList().size());
        Mat out = new Mat();
        MatOfDMatch mats = new MatOfDMatch();
        mats.fromList(good_matches);
        Features2d.drawMatches(m2, points2, m, points,mats , out);
        Highgui.imwrite("/Users/chintan/Downloads/one2.jpg", out);   
    }
    
}
