/*
Copyright (c) 2018 Uber Technologies, Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef MAZE_H
#define MAZE_H
#include <vector>
#include <fstream>
#include <iostream>
#include <math.h>
using namespace std;


namespace maze
{
    static bool disable=true;

    void inline scale1(float &v, float min, float max)
    {
        v= (v-min)/(max-min);
    }

    //for tracking the position of the navigator
    class position_accumulator
    {
    public:
        long size;
        long count;
        float* buffer;
        float cap;
        float minx,miny;
        float maxx,maxy;

        vector<int> dim;

        position_accumulator(vector<int> dims,
                            float _minx,float _miny,float _maxx, float _maxy):
            minx(_minx),miny(_miny),maxx(_maxx),maxy(_maxy)
        {
            count=0;
            dim=dims;
            size=1;
            cap=30;
            for(int x=0; x<dim.size(); x++)
                size*=dim[x];

            buffer=(float*)malloc(sizeof(float)*size);


            for(int x=0; x<size; x++)
                buffer[x]=0.0f;
        }

        void add_point(float* v)
        {
            long int index=0;
            long int multiplier=1;

            scale1(v[0],minx,maxx);
            scale1(v[1],miny,maxy);

            for(int x=0; x<dim.size(); x++)
            {
                int loc_index = v[x] * dim[x]; //+ 0.499;
                index+=multiplier*loc_index;
                multiplier*=dim[x];
            }
            if(index>=size)
                cout << "whoops..." << endl;
            if(buffer[index]<cap)
                buffer[index]++;
            count++;
        }

        void transform()
        {
            for(int x=0; x<size; x++)
                buffer[x]/=(float)count;
        }

        double entropy()
        {
            double ent=0.0;
            for(int x=0;x<size;x++)
            {
            if (buffer[x]!=0.0) {
            ent+=buffer[x]*log(buffer[x]);
        //ent+= -1;
            }

            }
            return -ent;
        }
        ~position_accumulator()
        {
            free(buffer);
        }

    };


    //simple point class
    class Point
    {
    public:
        Point(float x1,float y1)
        {
            x=x1;
            y=y1;
        }

        Point()
        {
        }

        Point(const Point& k)
        {
            x=k.x;
            y=k.y;
        }

        void fromfile(ifstream& file)
        {
            file >> x;
            file >> y;
        }

        //determine angle of vector defined by (0,0)->This Point
        float angle()
        {
            if(x==0.0)
            {
                if(y>0.0) return 90.0;
                return 270.0;
            }
            float ang=atan(y/x)/3.1415926*180.0;

            if(isnan(ang))
                cout << "NAN in angle\n";
            //quadrant 1 or 4
            if(x>0.0)
            {
                return ang;
            }
            return ang+180.0;
        }

        //rotate this point around another point
        void rotate(float angle,Point p)
        {
            float rad=angle/180.0*3.1415926;
            x-=p.x;
            y-=p.y;

            float ox=x;
            float oy=y;
            x=cos(rad)*ox-sin(rad)*oy;
            y=sin(rad)*ox+cos(rad)*oy;

            x+=p.x;
            y+=p.y;
        }
        //distance between this point and another point
        float distance(Point b)
        {
            float dx=b.x-x;
            float dy=b.y-y;
            return sqrt(dx*dx+dy*dy);
        }
        float x;
        float y;
    };

    //simple line segment class, used for maze walls
    class Line
    {
    public:
        Line(Point k,Point j)
        {
            a.x=k.x;
            a.y=k.y;
            b.x=j.x;
            b.y=j.y;
        }
        Line(ifstream& file)
        {
            a.fromfile(file);
            b.fromfile(file);
        }
        Line()
        {
        }
        //midpoint of the line segment
        Point midpoint()
        {
            Point newpoint;
            newpoint.x=(a.x+b.x)/2.0;
            newpoint.y=(a.y+b.y)/2.0;
            return newpoint;
        }

        //return point of intersection between two line segments if it exists
        Point intersection(Line L,bool &found)
        {

            Point pt(0.0,0.0);
            Point A(a);
            Point B(b);
            Point C(L.a);
            Point D(L.b);


            float rTop = (A.y-C.y)*(D.x-C.x)-(A.x-C.x)*(D.y-C.y);
            float rBot = (B.x-A.x)*(D.y-C.y)-(B.y-A.y)*(D.x-C.x);

            float sTop = (A.y-C.y)*(B.x-A.x)-(A.x-C.x)*(B.y-A.y);
            float sBot = (B.x-A.x)*(D.y-C.y)-(B.y-A.y)*(D.x-C.x);

            if ( (rBot == 0) || (sBot == 0))
            {
                //lines are parallel
                found = false;
                return pt;
            }

            float r = rTop/rBot;
            float s = sTop/sBot;

            if( (r > 0) && (r < 1) && (s > 0) && (s < 1) )
            {

                pt.x = A.x + r * (B.x - A.x);
                pt.y = A.y + r * (B.y - A.y);

                found=true;
                return pt;
            }

            else
            {

                found=false;
                return pt;
            }

        }

        //distance between line segment and point
        float distance(Point n)
        {
            float utop = (n.x-a.x)*(b.x-a.x)+(n.y-a.y)*(b.y-a.y);
            float ubot = a.distance(b);
            ubot*=ubot;
            if(ubot==0.0)
            {
                //cout << "Ubot zero?" << endl;
                return 0.0;
            }
            float u = utop/ubot;

            if(u<0 || u>1)
            {
                float d1=a.distance(n);
                float d2=b.distance(n);
                if(d1<d2) return d1;
                return d2;
            }
            Point p;
            p.x=a.x+u*(b.x-a.x);
            p.y=a.y+u*(b.y-a.y);
            return p.distance(n);
        }

        //line segment length
        float length()
        {
            return a.distance(b);
        }
        Point a;
        Point b;
    };

    //class for the maze navigator
    class Character
    {
    public:
        vector<float> rangeFinderAngles; //angles of range finder sensors
        vector<float> radarAngles1; //beginning angles for radar sensors
        vector<float> radarAngles2; //ending angles for radar sensors

        vector<float> radar; //stores radar outputs
        vector<float> poi_radar; //stores poi radar
        vector<float> rangeFinders; //stores rangefinder outputs
        Point location;
            Point start;
        bool collide;
        int collisions;
        float total_spin;
        float heading;
        float speed;
        float ang_vel;
        float radius;
        float rangefinder_range;

        float total_dist;
        void reset()
        {
            total_dist=0.0f;
            total_spin=0.0f;
            collide=false;
            collisions=0;
            heading=0.0f;
            speed=0.0f;
            ang_vel=0.0f;
            location = start;
        }

        Character()
        {
            total_dist=0.0f;
            total_spin=0.0f;
            collide=false;
            collisions=0;
            heading=0.0f;
            speed=0.0f;
            ang_vel=0.0f;
            radius=8.0f;
            rangefinder_range=100.0f;

            //define the range finder sensors
    //#define SIMPLE_SENSORS //for use with evolvability-inevitability
    #ifdef SIMPLE_SENSORS
            rangeFinderAngles.push_back(-30.0f);
            rangeFinderAngles.push_back(30.0f);
    #else

            rangeFinderAngles.push_back(-90.0f);
            rangeFinderAngles.push_back(-45.0f);
            rangeFinderAngles.push_back(0.0f);
            rangeFinderAngles.push_back(45.0f);
            rangeFinderAngles.push_back(90.0f);
            rangeFinderAngles.push_back(-180.0f);
    #endif

            //define the radar sensors
            radarAngles1.push_back(315.0);
            radarAngles2.push_back(405.0);

            radarAngles1.push_back(45.0);
            radarAngles2.push_back(135.0);

            radarAngles1.push_back(135.0);
            radarAngles2.push_back(225.0);

            radarAngles1.push_back(225.0);
            radarAngles2.push_back(315.0);

            for(int i=0; i<(int)rangeFinderAngles.size(); i++)
                rangeFinders.push_back(0.0);

            for(int i=0; i<(int)radarAngles1.size(); i++)
            {
                radar.push_back(0.0);
                poi_radar.push_back(0.0);
            }
        }



    };

    //all-encompassing environment class
    class Environment
    {
    public:
        Environment(const Environment &e)
        {
            steps=e.steps;
            hero.location = e.hero.location;
                    hero.start = e.hero.start;
            hero.heading = e.hero.heading;
            hero.speed=e.hero.speed;
            hero.ang_vel=e.hero.ang_vel;
            end=e.end;
            poi=e.poi;
            goalattract=e.goalattract;
            for(int i=0; i<(int)e.lines.size(); i++)
            {
                Line* x=new Line(*(e.lines[i]));
                lines.push_back(x);
            }
            update_rangefinders(hero);
            update_radar(hero);
            reachgoal=e.reachgoal;
            reachpoi=e.reachpoi;
            closest_to_poi = e.closest_to_poi;
            closest_to_target = e.closest_to_target;
        }

        int get_line_count() {
        return lines.size();
        }
        Line get_line(int idx) {
        Line* l = lines[idx];
        return Line(*l);
        }

        void get_range(float &minx,float &miny, float &maxx, float& maxy)
        {
            minx= 100000;
            miny= 100000;
            maxx = -1000000;
            maxy = -1000000;
            for(int x=0; x<lines.size(); x++)
            {
                if (lines[x]->a.x < minx)
                    minx = lines[x]->a.x;
                if (lines[x]->a.x > maxx)
                    maxx = lines[x]->a.x;
                if (lines[x]->b.x < minx)
                    minx = lines[x]->b.x;
                if (lines[x]->b.x > maxx)
                    maxx = lines[x]->b.x;
                if (lines[x]->a.y < miny)
                    miny = lines[x]->a.y;
                if (lines[x]->a.y > maxy)
                    maxy = lines[x]->a.y;
                if (lines[x]->b.y < miny)
                    miny = lines[x]->b.y;
                if (lines[x]->b.y > maxy)
                    maxy = lines[x]->b.y;
            }
        }
        //initialize environment from maze file
        Environment(const char* filename) : Environment()
        {
            load_from(filename);
        }

        Environment()
        {

        }

        void reset()
        {
            hero.reset();
            //update sensors
            update_rangefinders(hero);
            update_radar(hero);
        }

        void load_from(const char* filename)
        {
            ifstream inpfile(filename);
            int num_lines;
            inpfile >> disable;
            inpfile >> steps;
            inpfile >> num_lines; //read in how many line segments
            hero.location.fromfile(inpfile); //read initial location
                hero.start.x=hero.location.x;
                hero.start.y=hero.location.y;
            inpfile >> hero.heading; //read initial heading
            end.fromfile(inpfile); //read goal location
            poi.fromfile(inpfile);
            reachpoi=0;
            reachgoal=0;
            closest_to_poi = 100000.0;
            closest_to_target = 100000.0;
            goalattract=true;
            //read in line segments
            for(int i=0; i<num_lines; i++)
            {
                Line* x=new Line(inpfile);
                lines.push_back(x);
            }
            //update sensors
            update_rangefinders(hero);
            update_radar(hero);
        }

        //debug function
        void display()
        {
            cout << "Hero: " << hero.location.x << " " << hero.location.y << endl;
            cout << "EndPoint: " << end.x << " " << end.y << endl;
            cout << "Lines:" << endl;
            for(int i=0; i<(int)lines.size(); i++)
            {
                cout << lines[i]->a.x << " " << lines[i]->a.y << " " << lines[i]->b.x << " " << lines[i]->b.y << endl;
            }
        }

            float distance_to_start() {
                float dist=hero.location.distance(hero.start);
                return dist;
                }
        //used for fitness calculations
        float distance_to_target()
        {
            float dist=hero.location.distance(end);
            if(isnan(dist))
            {
                cout << "NAN Distance error..." << endl;
                return 500.0;
            }
            if(dist<10.0  && !reachgoal) {
                reachgoal=1; //if within 5 units, success!
                //closest_to_poi=10000000;
            }
            else if (!goalattract) reachgoal=0; //must we
            //remain close?
            if(dist<closest_to_target)
                closest_to_target=dist;
            return dist;
        }

        float distance_to_poi()
        {
            float dist=hero.location.distance(poi);
            if(isnan(dist))
            {
                cout << "NAN Distance error..." << endl;
                return 500.0;
            }
            if(dist<closest_to_poi)
                closest_to_poi=dist;

            if(dist<10.0) reachpoi=1; //if within 10 units, success!
            return dist;
        }

        void generate_neural_inputs_wrapper(float* inputs, int size) {
                generate_neural_inputs(inputs);
        }

        //create neural net inputs from sensors
        void generate_neural_inputs(float* inputs)
        {
            //bias
            inputs[0]=(1.0);

            //rangefinders
            int i,j,k;
            for(i=0; i<(int)hero.rangeFinders.size(); i++)
            {
                inputs[1+i]=(hero.rangeFinders[i]/hero.rangefinder_range);
                if(isnan(inputs[1+i]))
                    cout << "NAN in inputs" << endl;
            }

            //radar
            for(j=0; j<(int)hero.radar.size(); j++)
            {
                inputs[i+j+1]=(hero.radar[j]);
                if(isnan(inputs[i+j]))
                    cout << "NAN in inputs" << endl;
            }

            double distance = distance_to_target();
            distance = 1.0 - (distance/100.0);
            if(distance<0.0)
                distance=0.0;
            //inputs[i+j]=distance;

        /*
            //poi radar
            for(k=0; k<(int)hero.poi_radar.size(); k++)
            {
                inputs[i+j+k+1]=(hero.poi_radar[k]);
                if(isnan(inputs[i+j+k]))
                    cout << "NaN in inputs" << endl;
            }


            distance = distance_to_poi();
            distance = 1.0 - (distance/100.0);
            if(distance<0.0)
                distance=0.0;
            //inputs[i+j+k+1] = distance;

            inputs[i+j+k+1] = reachgoal; //was reachpoi
        */

            return;
        }

        //transform neural net outputs into angular velocity and speed
        void interpret_outputs(float o1,float o2)
        {
            if(isnan(o1) || isnan(o2))
                cout << "OUTPUT ISNAN" << endl;
        if (o1>1.0) o1=1.0;
        if (o1<0.0) o1=0.0;
        if (o2>1.0) o2=1.0;
        if (o2<0.0) o2=0.0;

        bool do_accel = false;
            //why apply accelerations?
            if ( do_accel) {
            hero.ang_vel+=(o1-0.5)*1.0;
            hero.speed+=(o2-0.5)*1.0;
            }
            else {


            //why apply accelerations?
            //hero.ang_vel+=(o1-0.5)*1.0;
            //hero.speed+=(o2-0.5)*1.0;

            //if(o1<0.2)
            //  hero.ang_vel= -0.5;
            //if(o2>0.8)
            // hero.ang_vel= +0.5;
            //if(o3>0.5)
            //  hero.collide=true;
            //IF YOU WANT velocity instead of accel
        //


            float new_ang_vel=(o1-0.5)*6.0;
            float new_speed=(o2-0.5)*6.0;
            float d_ang = new_ang_vel-hero.ang_vel;
            float d_speed= new_speed-hero.speed;

            if(d_ang>=0.2) d_ang=0.2;
            if(d_ang<=-0.2) d_ang=-0.2;
            if(d_speed>=0.2) d_speed=0.2;
            if(d_speed<=-0.2) d_speed=-0.2;
            hero.ang_vel+=d_ang; //(o1-0.5)*6.0;
            hero.speed+=d_speed; //(o2-0.5)*6.0;
        }
            hero.total_spin+=fabs(hero.ang_vel);
            //constraints of speed & angular velocity
            if(hero.speed>3.0) hero.speed=3.0;
            if(hero.speed<-3.0) hero.speed=(-3.0);
            if(hero.ang_vel>3.0) hero.ang_vel=3.0;
            if(hero.ang_vel<-3.0) hero.ang_vel=(-3.0);
        }

        //run a time step of the simulation
        void Update()
        {
            //	if (reachgoal && goalattract)
            //		return;
            float vx=cos(hero.heading/180.0*3.1415926)*hero.speed;
            float vy=sin(hero.heading/180.0*3.1415926)*hero.speed;
            if(isnan(vx))
                cout << "VX NAN" << endl;

            hero.heading+=hero.ang_vel;
            if(isnan(hero.ang_vel))
                cout << "HERO ANG VEL NAN" << endl;

            if(hero.heading>360) hero.heading-=360;
            if(hero.heading<0) hero.heading+=360;

            Point newloc;
            newloc.x=vx+hero.location.x;
            newloc.y=vy+hero.location.y;
            //collision detection
            if(!hero.collide && !collide_lines(newloc,hero.radius))
            {
                hero.location.x=newloc.x;
                hero.location.y=newloc.y;
            hero.total_dist+=fabs(hero.speed);
            }
            else
            {
                hero.collisions++;
                if(disable)
                    hero.collide=true;
            }
            update_rangefinders(hero);
            update_radar(hero);
        }

        //see if navigator has hit anything
        bool collide_lines(Point loc,float rad)
        {
            for(int i=0; i<(int)lines.size(); i++)
            {
                if(lines[i]->distance(loc)<rad)
                    return true;
            }
            return false;
        }

        int get_sensor_size() {
            return 1+hero.rangeFinderAngles.size()+hero.radar.size();
        }

        //rangefinder sensors
        void update_rangefinders(Character& h)
        {
            //iterate through each sensor
            for(int i=0; i<(int)h.rangeFinderAngles.size(); i++)
            {
                float rad=h.rangeFinderAngles[i]/180.0*3.1415926; //radians...

                //project a point from the hero's location outwards
                Point proj_point(h.location.x+cos(rad)*h.rangefinder_range,
                                h.location.y+sin(rad)*h.rangefinder_range);

                //rotate the project point by the hero's heading
                proj_point.rotate(h.heading,h.location);

                //create a line segment from the hero's location to projected
                Line projected_line(h.location,proj_point);

                float range=h.rangefinder_range; //set range to max by default

                //now test against the environment to see if we hit anything
                for(int j=0; j<(int)lines.size(); j++)
                {
                    bool found=false;
                    Point intersection=lines[j]->intersection(projected_line,found);
                    if(found)
                    {
                        //if so, then update the range to the distance
                        float found_range = intersection.distance(h.location);

                        //we want the closest intersection
                        if(found_range<range)
                            range=found_range;
                    }
                }
                if(isnan(range))
                    cout << "RANGE NAN" << endl;
                h.rangeFinders[i]=range;
            }
        }

        void update_radar(Character& h)
        {
            update_radar_gen(h,end,h.radar);
            if(true) //if(!reachpoi)
                update_radar_gen(h,poi,h.poi_radar);
            else
            {
                for(int x=0; x<h.poi_radar.size(); x++)
                    h.poi_radar[x]=0.0;
            }
        }
        //radar sensors
        void update_radar_gen(Character& h,Point target, vector<float>& radar_arr)
        {
            //possible optimization for mc, move later, avoid sqrt
            double distance = h.location.distance(target);
            distance = 1.0 - (distance/200.0);
            if(distance<0.2)
                distance=0.2;
            bool compass=false; //if you want it to be compass instead of
            //target indicator
            if(compass) {
                target.x=h.location.x;
                target.y=h.location.y-50;
            }
            //rotate goal with respect to heading of navigator
            target.rotate(-h.heading,h.location);

            //translate with respect to location of navigator
            target.x-=h.location.x;
            target.y-=h.location.y;

            //what angle is the vector between target & navigator
            float angle=target.angle();

            //fire the appropriate radar sensor
            for(int i=0; i<(int)h.radarAngles1.size(); i++)
            {
                radar_arr[i]=0.0;

                if(angle>=h.radarAngles1[i] && angle<h.radarAngles2[i])
                    radar_arr[i]=1.0; //distance;

                if(angle+360.0>=h.radarAngles1[i] && angle+360.0<h.radarAngles2[i])
                    radar_arr[i]=1.0;//distance;
            }
        }

        ~Environment()
        {
            //clean up lines!
            for(int i=0; i<(int)lines.size(); i++)
                delete lines[i];
        }
        double closest_to_target;
        int steps;
        double closest_to_poi;
        vector<Line*> lines; //maze line segments
        Character hero; //navigator
        Point end; //the goal
        Point poi; //point of interest
        int reachpoi;
        int reachgoal;
        bool goalattract;
    };
}

#endif
