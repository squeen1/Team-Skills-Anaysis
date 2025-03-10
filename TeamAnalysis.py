import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

class EnhancedTechnicalDashboard:
    def __init__(self, skills_csv_path, certifications_csv_path=None):
        """
        Initialize dashboard by reading CSV files
        
        Args:
            skills_csv_path (str): Path to the CSV file containing team skills data
            certifications_csv_path (str): Path to the CSV file containing certification data
        """
        # Read the skills CSV file
        self.skills_df = pd.read_csv(skills_csv_path)
        
        # Extract engineer names (from the specified column with actual names)
        self.engineers = self.skills_df.iloc[8:16, 1].tolist()
        
        # Extract technologies (from row 6, columns P-Z)
        self.technologies = self.skills_df.iloc[6, 15:26].tolist()
        
        # Extract skill matrix (rows 8-16, columns P-Z)
        skill_matrix = self.skills_df.iloc[8:16, 15:26].copy()
        
        # Custom numeric conversion
        def safe_numeric_convert(value):
            try:
                # Try to convert to float
                return float(value)
            except (ValueError, TypeError):
                try:
                    # Try to extract numeric part
                    numeric_part = re.findall(r'\d+', str(value))
                    return float(numeric_part[0]) if numeric_part else 0
                except:
                    # If all else fails, return 0
                    return 0
        
        # Convert matrix to numeric
        skill_matrix = skill_matrix.apply(lambda col: col.map(safe_numeric_convert))
        
        # Create a DataFrame for easier manipulation
        self.skill_df = pd.DataFrame(
            skill_matrix.values, 
            index=self.engineers, 
            columns=self.technologies
        )
        
        # Ensure numeric type
        self.skill_df = self.skill_df.astype(float)

        # Initialize certification dataframe
        self.cert_df = None
        
        # Load certification data if provided
        if certifications_csv_path:
            self.load_certification_data(certifications_csv_path)

    def load_certification_data(self, certifications_csv_path):
        """
        Load certification data from CSV file
        
        Args:
            certifications_csv_path (str): Path to the CSV file containing certification data
        """
        # Read certification data
        cert_data = pd.read_csv(certifications_csv_path)
        
        # Process and restructure certification data
        # Assuming format: Engineer, Certification, Status (Obtained/Expired/In Progress), Date
        self.cert_df = cert_data
        
        # Create pivot table for easier analysis
        # Engineers as rows, certifications as columns, values as status
        self.cert_pivot = pd.pivot_table(
            cert_data,
            index='Engineer',
            columns='Certification',
            values='Status',
            aggfunc='first',
            fill_value='None'
        )
        
    def skill_level_mapping(self, score):
        """
        Map numerical skill scores to descriptive levels
        """
        skill_levels = {
            1: 'Novice',
            2: 'Advanced Beginner',
            3: 'Competent',
            4: 'Proficient',
            5: 'Expert'
        }
        return skill_levels.get(int(score), 'Unknown')
    
    def cert_status_color_mapping(self):
        """
        Return color mapping for certification statuses
        """
        return {
            'Obtained': 'green',
            'In Progress': 'yellow',
            'Expired': 'orange',
            'None': 'lightgrey'
        }
    
    def technology_coverage_heatmap(self):
        """
        Create a heatmap showing team's technical skills
        """
        plt.figure(figsize=(15, 10))
        
        # Create heatmap with numerical scores
        sns.heatmap(
            self.skill_df, 
            annot=True, 
            cmap='YlGnBu', 
            fmt='.0f',
            cbar_kws={'label': 'Skill Level (1-5)'}
        )
        
        plt.title('Team Technical Skill Coverage', fontsize=16)
        plt.xlabel('Technologies', fontsize=12)
        plt.ylabel('Engineers', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('technology_coverage_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def certification_coverage_heatmap(self):
        """
        Create a heatmap showing team's certification status
        """
        if self.cert_df is None:
            print("No certification data available")
            return
        
        plt.figure(figsize=(15, 10))
        
        # Create a custom colormap for certification status
        status_colors = self.cert_status_color_mapping()
        status_values = list(status_colors.keys())
        
        # Create a mapping from status to numeric value for plotting
        status_to_num = {status: i for i, status in enumerate(status_values)}
        
        # Convert statuses to numeric values
        cert_matrix_numeric = self.cert_pivot.applymap(lambda x: status_to_num.get(x, 0))
        
        # Create custom colormap
        cmap = ListedColormap([status_colors[status] for status in status_values])
        
        # Create heatmap
        sns.heatmap(
            cert_matrix_numeric,
            annot=self.cert_pivot.values,
            cmap=cmap,
            fmt='',
            cbar=False
        )
        
        # Create legend
        patches = [mpatches.Patch(color=color, label=status) for status, color in status_colors.items()]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.title('Team Certification Coverage', fontsize=16)
        plt.xlabel('Certifications', fontsize=12)
        plt.ylabel('Engineers', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('certification_coverage_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def skill_distribution_analysis(self):
        """
        Analyze skill distribution across technologies
        """
        # Calculate average skill level per technology
        avg_skills = self.skill_df.mean()
        
        plt.figure(figsize=(12, 6))
        avg_skills.plot(kind='bar')
        plt.title('Average Skill Level by Technology', fontsize=16)
        plt.xlabel('Technologies', fontsize=12)
        plt.ylabel('Average Skill Level', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('technology_avg_skills.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detailed skill level distribution
        skill_distribution = {}
        for technology in self.technologies:
            tech_skills = self.skill_df[technology]
            skill_counts = {
                'Novice': sum(tech_skills == 1),
                'Advanced Beginner': sum(tech_skills == 2),
                'Competent': sum(tech_skills == 3),
                'Proficient': sum(tech_skills == 4),
                'Expert': sum(tech_skills == 5)
            }
            skill_distribution[technology] = skill_counts
        
        return skill_distribution
    
    def certification_analysis(self):
        """
        Analyze certification distribution across the team
        """
        if self.cert_df is None:
            print("No certification data available")
            return None
        
        # Count certifications by status
        cert_status_counts = self.cert_df.groupby(['Certification', 'Status']).size().unstack(fill_value=0)
        
        # Plot certification status distribution
        plt.figure(figsize=(12, 8))
        cert_status_counts.plot(kind='bar', stacked=True)
        plt.title('Certification Status Distribution', fontsize=16)
        plt.xlabel('Certification', fontsize=12)
        plt.ylabel('Number of Team Members', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Status')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('certification_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calculate percentage of team with each certification
        team_size = len(self.engineers)
        cert_coverage = {}
        
        for cert in self.cert_df['Certification'].unique():
            # Count people with this certification (obtained or in progress)
            obtained = len(self.cert_df[(self.cert_df['Certification'] == cert) & 
                                      (self.cert_df['Status'] == 'Obtained')])
            in_progress = len(self.cert_df[(self.cert_df['Certification'] == cert) & 
                                         (self.cert_df['Status'] == 'In Progress')])
            expired = len(self.cert_df[(self.cert_df['Certification'] == cert) & 
                                     (self.cert_df['Status'] == 'Expired')])
            
            cert_coverage[cert] = {
                'Obtained': obtained,
                'Obtained_Percentage': round(obtained / team_size * 100, 1),
                'In Progress': in_progress,
                'In_Progress_Percentage': round(in_progress / team_size * 100, 1),
                'Expired': expired,
                'Expired_Percentage': round(expired / team_size * 100, 1),
                'No Certification': team_size - (obtained + in_progress + expired),
                'No_Certification_Percentage': round((team_size - (obtained + in_progress + expired)) / team_size * 100, 1)
            }
        
        return cert_coverage
    
    def skill_cert_correlation(self):
        """
        Analyze correlation between skill levels and certifications
        """
        if self.cert_df is None:
            print("No certification data available")
            return None
        
        # Prepare data for correlation analysis
        # For each engineer and technology, look at their skill level and certification status
        
        # Get list of relevant certifications (those that match technologies)
        tech_certs = {}
        for tech in self.technologies:
            # Find certifications related to this technology
            # This is a simplified approach - in a real implementation, you'd need a more 
            # sophisticated mapping between technologies and certifications
            related_certs = [cert for cert in self.cert_pivot.columns if tech.lower() in cert.lower()]
            if related_certs:
                tech_certs[tech] = related_certs
        
        # Analysis results
        correlation_results = {}
        
        for tech, certs in tech_certs.items():
            tech_skills = self.skill_df[tech]
            
            for cert in certs:
                # Get certification status for each engineer
                cert_statuses = []
                for engineer in self.engineers:
                    try:
                        status = self.cert_pivot.loc[engineer, cert]
                    except (KeyError, ValueError):
                        status = 'None'
                    cert_statuses.append(status)
                
                # Convert status to numeric for correlation
                # Obtained=3, In Progress=2, Expired=1, None=0
                status_map = {'Obtained': 3, 'In Progress': 2, 'Expired': 1, 'None': 0}
                cert_status_numeric = [status_map.get(status, 0) for status in cert_statuses]
                
                # Calculate average skill level for each certification status
                skills_by_status = {}
                for status in ['Obtained', 'In Progress', 'Expired', 'None']:
                    skills = [skill for skill, s in zip(tech_skills, cert_statuses) if s == status]
                    if skills:
                        skills_by_status[status] = sum(skills) / len(skills)
                    else:
                        skills_by_status[status] = 0
                
                correlation_results[f"{tech} - {cert}"] = skills_by_status
        
        return correlation_results
    
    def generate_detailed_report(self):
        """
        Generate a comprehensive skill and certification analysis report
        """
        # Individual engineer skill summary
        individual_summary = pd.DataFrame(index=self.engineers)
        
        for technology in self.technologies:
            individual_summary[technology] = self.skill_df[technology].apply(self.skill_level_mapping)
        
        # Overall team skill statistics
        team_stats = {
            'Total Technologies Covered': len(self.technologies),
            'Average Team Skill Level': self.skill_df.values.mean(),
            'Highest Skilled Technology': self.skill_df.mean().idxmax(),
            'Lowest Skilled Technology': self.skill_df.mean().idxmin()
        }
        
        # Add certification statistics if available
        if self.cert_df is not None:
            cert_counts = self.cert_df.groupby('Status').size()
            
            cert_stats = {
                'Total Certifications Tracked': len(self.cert_df['Certification'].unique()),
                'Certifications Obtained': cert_counts.get('Obtained', 0),
                'Certifications In Progress': cert_counts.get('In Progress', 0),
                'Certifications Expired': cert_counts.get('Expired', 0),
                'Most Common Certification': self.cert_df['Certification'].value_counts().idxmax(),
                'Certification Density': round(len(self.cert_df) / len(self.engineers), 2)
            }
            
            team_stats.update(cert_stats)
            
            # Add certification data to individual summary
            if len(self.cert_pivot) > 0:
                for cert in self.cert_pivot.columns:
                    # Only include engineers present in both datasets
                    for engineer in self.engineers:
                        if engineer in self.cert_pivot.index:
                            individual_summary.loc[engineer, f"Cert: {cert}"] = self.cert_pivot.loc[engineer, cert]
                        else:
                            individual_summary.loc[engineer, f"Cert: {cert}"] = 'None'
        
        # Export reports
        individual_summary.to_csv('individual_skill_cert_summary.csv')
        
        with open('team_skill_cert_stats.txt', 'w') as f:
            for stat, value in team_stats.items():
                f.write(f"{stat}: {value}\n")
        
        return {
            'individual_summary': individual_summary,
            'team_stats': team_stats
        }
    
    def generate_skill_gap_analysis(self):
        """
        Generate analysis of skill gaps in the team
        """
        # Calculate team coverage for each technology
        tech_coverage = {}
        
        for tech in self.technologies:
            # Count number of people at each skill level
            skill_counts = self.skill_df[tech].value_counts().to_dict()
            
            # Calculate percentage of team at each skill level
            team_size = len(self.engineers)
            skill_percentages = {level: count / team_size * 100 for level, count in skill_counts.items()}
            
            # Calculate percentage of team at competent level or above (3+)
            competent_plus = sum(count for level, count in skill_counts.items() if level >= 3)
            competent_percentage = competent_plus / team_size * 100
            
            # Determine if this is a skill gap area
            is_gap = competent_percentage < 50  # less than 50% of team is competent or above
            
            tech_coverage[tech] = {
                'skill_counts': skill_counts,
                'skill_percentages': skill_percentages,
                'competent_plus_count': competent_plus,
                'competent_plus_percentage': competent_percentage,
                'is_gap': is_gap
            }
        
        # Identify critical skill gaps
        critical_gaps = {tech: data for tech, data in tech_coverage.items() if data['is_gap']}
        
        # Visualize skill gaps
        plt.figure(figsize=(12, 8))
        
        # Plot competency percentages
        competency_data = {tech: data['competent_plus_percentage'] for tech, data in tech_coverage.items()}
        tech_series = pd.Series(competency_data)
        
        # Sort the data
        tech_series = tech_series.sort_values(ascending=True)
        
        # Create bar colors based on threshold
        colors = ['red' if val < 50 else 'green' for val in tech_series.values]
        
        tech_series.plot(kind='barh', color=colors)
        plt.axvline(x=50, color='red', linestyle='--')
        plt.title('Technology Competency Coverage (% of Team at Level 3+)', fontsize=16)
        plt.xlabel('Percentage of Team', fontsize=12)
        plt.ylabel('Technology', fontsize=12)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('skill_gap_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'tech_coverage': tech_coverage,
            'critical_gaps': critical_gaps
        }
    
    def generate_training_recommendations(self):
        """
        Generate training recommendations based on skill gaps and certifications
        """
        # Get skill gap analysis
        gap_analysis = self.generate_skill_gap_analysis()
        
        # Prepare recommendations
        recommendations = []
        
        # For each critical gap, identify engineers to train
        for tech, data in gap_analysis['critical_gaps'].items():
            # Find engineers with skill level 2 (Advanced Beginner)
            potential_trainees = []
            
            for engineer in self.engineers:
                skill_level = self.skill_df.loc[engineer, tech]
                
                if skill_level == 2:  # Advanced Beginner - most ready to advance to Competent
                    potential_trainees.append(engineer)
            
            # Add recommendation
            recommendation = {
                'technology': tech,
                'gap_percentage': 50 - data['competent_plus_percentage'],  # how far below 50%
                'potential_trainees': potential_trainees,
                'recommended_action': f"Provide training for {len(potential_trainees)} team members to address {tech} skill gap"
            }
            
            recommendations.append(recommendation)
        
        # Add certification recommendations if cert data available
        if self.cert_df is not None:
            # For each critical gap, identify relevant certifications
            for tech, data in gap_analysis['critical_gaps'].items():
                # Find certifications related to this technology
                related_certs = [cert for cert in self.cert_df['Certification'].unique() 
                               if tech.lower() in cert.lower()]
                
                if related_certs:
                    # Find engineers who don't have these certifications
                    for cert in related_certs:
                        engineers_without_cert = []
                        
                        for engineer in self.engineers:
                            has_cert = False
                            
                            # Check if engineer has this certification
                            engineer_certs = self.cert_df[self.cert_df['Engineer'] == engineer]
                            if not engineer_certs.empty:
                                has_cert = any((engineer_certs['Certification'] == cert) & 
                                              (engineer_certs['Status'].isin(['Obtained', 'In Progress'])))
                            
                            if not has_cert:
                                engineers_without_cert.append(engineer)
                        
                        # Add recommendation
                        if engineers_without_cert:
                            recommendation = {
                                'technology': tech,
                                'certification': cert,
                                'engineers_without_cert': engineers_without_cert,
                                'recommended_action': f"Encourage {len(engineers_without_cert)} team members to pursue {cert} certification"
                            }
                            
                            recommendations.append(recommendation)
        
        # Save recommendations to file
        with open('training_recommendations.txt', 'w') as f:
            f.write("TRAINING RECOMMENDATIONS\n")
            f.write("======================\n\n")
            
            for i, rec in enumerate(recommendations, 1):
                f.write(f"Recommendation {i}:\n")
                f.write(f"Technology: {rec['technology']}\n")
                
                if 'gap_percentage' in rec:
                    f.write(f"Gap: {rec['gap_percentage']:.1f}% below target\n")
                
                if 'certification' in rec:
                    f.write(f"Certification: {rec['certification']}\n")
                
                f.write(f"Action: {rec['recommended_action']}\n")
                
                if 'potential_trainees' in rec:
                    f.write("Potential trainees: " + ", ".join(rec['potential_trainees']) + "\n")
                
                if 'engineers_without_cert' in rec:
                    f.write("Engineers without certification: " + ", ".join(rec['engineers_without_cert']) + "\n")
                
                f.write("\n")
        
        return recommendations
    



    def generate_individual_development_plans(self):
        """
        Generate personalized development plans for each team member
        based on their skills, certifications, and team needs
        """
        # Ensure we have gap analysis data
        gap_analysis = self.generate_skill_gap_analysis()
        critical_gaps = gap_analysis['critical_gaps']
        
        # Create directory for individual plans if it doesn't exist
        import os
        if not os.path.exists('development_plans'):
            os.makedirs('development_plans')
        
        # Summary of all plans for reporting
        all_plans_summary = {}
        
        # Generate a plan for each engineer
        for engineer in self.engineers:
            # Collect engineer's current skill levels
            current_skills = {}
            for tech in self.technologies:
                skill_level = self.skill_df.loc[engineer, tech]
                skill_name = self.skill_level_mapping(skill_level)
                current_skills[tech] = {
                    'level': skill_level,
                    'description': skill_name
                }
            
            # Collect engineer's certifications if available
            current_certs = []
            planned_certs = []
            expired_certs = []
            if self.cert_df is not None:
                engineer_certs = self.cert_df[self.cert_df['Engineer'] == engineer]
                
                for _, cert_row in engineer_certs.iterrows():
                    cert_info = {
                        'name': cert_row['Certification'],
                        'date': cert_row['Date']
                    }
                    
                    if cert_row['Status'] == 'Obtained':
                        current_certs.append(cert_info)
                    elif cert_row['Status'] == 'In Progress':
                        planned_certs.append(cert_info)
                    elif cert_row['Status'] == 'Expired':
                        expired_certs.append(cert_info)
            
            # Identify development priorities (3 categories)
            critical_priorities = []  # Urgent team gaps where this person can help
            growth_priorities = []    # Areas where the engineer is close to leveling up
            cert_priorities = []      # Certification opportunities
            
            # 1. Critical priorities: Team gaps where this person can help (level 2)
            for tech, data in critical_gaps.items():
                if current_skills[tech]['level'] == 2:  # Advanced Beginner
                    critical_priorities.append({
                        'technology': tech,
                        'current_level': current_skills[tech]['description'],
                        'target_level': 'Competent',
                        'reason': 'Critical team skill gap'
                    })
            
            # 2. Growth priorities: Technologies where engineer is close to next level
            for tech, skill_data in current_skills.items():
                # Check if this is already a critical priority
                if any(p['technology'] == tech for p in critical_priorities):
                    continue
                    
                # Focus on advancing engineers to the next level
                # Prioritize progression to competent (level 3) and proficient (level 4)
                if skill_data['level'] == 2:  # Advanced Beginner -> Competent
                    growth_priorities.append({
                        'technology': tech,
                        'current_level': skill_data['description'],
                        'target_level': 'Competent',
                        'reason': 'Ready to advance to competency'
                    })
                elif skill_data['level'] == 3:  # Competent -> Proficient
                    growth_priorities.append({
                        'technology': tech,
                        'current_level': skill_data['description'],
                        'target_level': 'Proficient',
                        'reason': 'Opportunity to develop expertise'
                    })
            
            # 3. Certification priorities
            if self.cert_df is not None:
                # Find relevant certifications for this engineer's skill areas
                for tech, skill_data in current_skills.items():
                    # Only recommend certs for areas where they have some skill (level 2+)
                    if skill_data['level'] >= 2:
                        # Find related certifications they don't already have
                        if len(self.cert_df) > 0:
                            related_certs = [cert for cert in self.cert_df['Certification'].unique() 
                                        if tech.lower() in cert.lower()]
                            
                            for cert in related_certs:
                                # Check if they already have this cert or it's in progress
                                has_cert = any(c['name'] == cert for c in current_certs)
                                in_progress = any(c['name'] == cert for c in planned_certs)
                                
                                if not has_cert and not in_progress:
                                    cert_priorities.append({
                                        'certification': cert,
                                        'related_technology': tech,
                                        'current_tech_level': skill_data['description'],
                                        'reason': 'Validates and enhances existing skills'
                                    })
            
            # Special case: Recommend renewing expired certifications
            for cert in expired_certs:
                cert_priorities.append({
                    'certification': cert['name'],
                    'related_technology': 'N/A',
                    'current_tech_level': 'N/A',
                    'reason': 'Certification renewal needed',
                    'expiry_date': cert['date']
                })
            
            # Create a development timeline with short and long-term goals
            timeline = {
                'immediate': [],  # 0-3 months
                'short_term': [], # 3-6 months
                'long_term': []   # 6-12 months
            }
            
            # Assign priorities to timeline
            # Critical priorities go to immediate
            for priority in critical_priorities[:2]:  # Limit to top 2
                timeline['immediate'].append({
                    'type': 'skill',
                    'focus': priority['technology'],
                    'goal': f"Advance from {priority['current_level']} to {priority['target_level']}",
                    'reason': priority['reason']
                })
            
            # Expired certs to immediate
            for cert in expired_certs:
                timeline['immediate'].append({
                    'type': 'certification',
                    'focus': cert['name'],
                    'goal': "Renew expired certification",
                    'reason': "Maintain validated credentials"
                })
            
            # In-progress certs to immediate/short-term based on date
            import datetime
            current_date = datetime.datetime.now()
            for cert in planned_certs:
                try:
                    cert_date = datetime.datetime.strptime(cert['date'], '%Y-%m-%d')
                    months_away = (cert_date.year - current_date.year) * 12 + (cert_date.month - current_date.month)
                    
                    if months_away <= 3:
                        timeline['immediate'].append({
                            'type': 'certification',
                            'focus': cert['name'],
                            'goal': "Complete certification in progress",
                            'reason': "Already investing in this credential"
                        })
                    else:
                        timeline['short_term'].append({
                            'type': 'certification',
                            'focus': cert['name'],
                            'goal': "Complete certification in progress",
                            'reason': "Already investing in this credential"
                        })
                except:
                    # If date format is invalid, default to short-term
                    timeline['short_term'].append({
                        'type': 'certification',
                        'focus': cert['name'],
                        'goal': "Complete certification in progress",
                        'reason': "Already investing in this credential"
                    })
            
            # Growth priorities to short/long term
            for i, priority in enumerate(growth_priorities):
                if i < 2:  # Top 2 to short-term
                    timeline['short_term'].append({
                        'type': 'skill',
                        'focus': priority['technology'],
                        'goal': f"Advance from {priority['current_level']} to {priority['target_level']}",
                        'reason': priority['reason']
                    })
                else:  # Rest to long-term
                    timeline['long_term'].append({
                        'type': 'skill',
                        'focus': priority['technology'],
                        'goal': f"Advance from {priority['current_level']} to {priority['target_level']}",
                        'reason': priority['reason']
                    })
            
            # New cert priorities to long term
            for i, priority in enumerate(cert_priorities):
                if 'expiry_date' in priority:  # Already handled expired certs
                    continue
                    
                if i < 2:  # Limit to top 2
                    timeline['long_term'].append({
                        'type': 'certification',
                        'focus': priority['certification'],
                        'goal': "Obtain new certification",
                        'reason': priority['reason']
                    })
            
            # Generate specific action items for each timeline entry
            actions = []
            
            for period, goals in timeline.items():
                for goal in goals:
                    if goal['type'] == 'skill':
                        # Generate skill-based actions
                        tech = goal['focus']
                        
                        # Find training options based on current level
                        current_level = None
                        for p in critical_priorities + growth_priorities:
                            if p['technology'] == tech:
                                current_level = p['current_level']
                                break
                        
                        if current_level == 'Advanced Beginner':
                            actions.append({
                                'timeline': period,
                                'focus': tech,
                                'action': f"Complete intermediate {tech} training course",
                                'resources': f"Online courses, internal workshops, practical exercises with {tech}"
                            })
                            actions.append({
                                'timeline': period,
                                'focus': tech,
                                'action': f"Work on a project using {tech} with mentoring",
                                'resources': "Mentor support, practice project, documentation"
                            })
                        elif current_level == 'Competent':
                            actions.append({
                                'timeline': period,
                                'focus': tech,
                                'action': f"Complete advanced {tech} training",
                                'resources': f"Advanced courses, technical deep dives, {tech} reference materials"
                            })
                            actions.append({
                                'timeline': period,
                                'focus': tech,
                                'action': f"Lead a small project or component involving {tech}",
                                'resources': "Project opportunity, peer review, advanced documentation"
                            })
                    elif goal['type'] == 'certification':
                        # Generate certification-based actions
                        cert = goal['focus']
                        
                        if "renew" in goal['goal'].lower():
                            actions.append({
                                'timeline': period,
                                'focus': cert,
                                'action': f"Complete renewal requirements for {cert}",
                                'resources': "Renewal documentation, refresher courses, exam preparation"
                            })
                        elif "progress" in goal['goal'].lower():
                            actions.append({
                                'timeline': period,
                                'focus': cert,
                                'action': f"Continue preparation for {cert} certification",
                                'resources': "Study materials, practice exams, certification prep course"
                            })
                        else:
                            actions.append({
                                'timeline': period,
                                'focus': cert,
                                'action': f"Begin preparation for {cert} certification",
                                'resources': "Study materials, training courses, certification guides"
                            })
            
            # Write the individual development plan to a file
            plan_filename = f"development_plans/IDP_{engineer.replace(' ', '_')}.txt"
            
            with open(plan_filename, 'w') as f:
                f.write(f"INDIVIDUAL DEVELOPMENT PLAN\n")
                f.write(f"==========================\n")
                f.write(f"Engineer: {engineer}\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n")
                
                # Current Skills Section
                f.write("CURRENT SKILL PROFILE\n")
                f.write("--------------------\n")
                for tech, data in sorted(current_skills.items(), 
                                        key=lambda x: x[1]['level'], 
                                        reverse=True):
                    f.write(f"{tech}: {data['description']} (Level {data['level']})\n")
                f.write("\n")
                
                # Current Certifications Section
                if current_certs or planned_certs or expired_certs:
                    f.write("CERTIFICATION STATUS\n")
                    f.write("-------------------\n")
                    
                    if current_certs:
                        f.write("Active Certifications:\n")
                        for cert in current_certs:
                            f.write(f"- {cert['name']} (Obtained: {cert['date']})\n")
                        f.write("\n")
                    
                    if planned_certs:
                        f.write("In-Progress Certifications:\n")
                        for cert in planned_certs:
                            f.write(f"- {cert['name']} (Target date: {cert['date']})\n")
                        f.write("\n")
                    
                    if expired_certs:
                        f.write("Expired Certifications (Renewal Recommended):\n")
                        for cert in expired_certs:
                            f.write(f"- {cert['name']} (Expired: {cert['date']})\n")
                        f.write("\n")
                
                # Development Timeline Section
                f.write("DEVELOPMENT TIMELINE\n")
                f.write("-------------------\n")
                
                f.write("Immediate Priorities (0-3 months):\n")
                if timeline['immediate']:
                    for goal in timeline['immediate']:
                        f.write(f"- {goal['goal']} for {goal['focus']}\n")
                        f.write(f"  Reason: {goal['reason']}\n")
                else:
                    f.write("- No immediate priorities identified\n")
                f.write("\n")
                
                f.write("Short-Term Goals (3-6 months):\n")
                if timeline['short_term']:
                    for goal in timeline['short_term']:
                        f.write(f"- {goal['goal']} for {goal['focus']}\n")
                        f.write(f"  Reason: {goal['reason']}\n")
                else:
                    f.write("- No short-term goals identified\n")
                f.write("\n")
                
                f.write("Long-Term Goals (6-12 months):\n")
                if timeline['long_term']:
                    for goal in timeline['long_term']:
                        f.write(f"- {goal['goal']} for {goal['focus']}\n")
                        f.write(f"  Reason: {goal['reason']}\n")
                else:
                    f.write("- No long-term goals identified\n")
                f.write("\n")
                
                # Action Plan Section
                f.write("ACTION PLAN\n")
                f.write("-----------\n")
                
                # Group actions by timeline
                for period in ['immediate', 'short_term', 'long_term']:
                    period_actions = [a for a in actions if a['timeline'] == period]
                    
                    if period_actions:
                        if period == 'immediate':
                            f.write("Immediate Actions (0-3 months):\n")
                        elif period == 'short_term':
                            f.write("Short-Term Actions (3-6 months):\n")
                        else:
                            f.write("Long-Term Actions (6-12 months):\n")
                        
                        for i, action in enumerate(period_actions, 1):
                            f.write(f"{i}. {action['action']}\n")
                            f.write(f"   Focus: {action['focus']}\n")
                            f.write(f"   Resources: {action['resources']}\n")
                            f.write("\n")
                
                # Summary section
                f.write("DEVELOPMENT SUMMARY\n")
                f.write("------------------\n")
                f.write(f"Primary focus areas: {', '.join(set([g['focus'] for p in timeline.values() for g in p[:2]]))}\n")
                
                critical_count = len(critical_priorities)
                if critical_count > 0:
                    f.write(f"Addressing {critical_count} critical team skill gap(s)\n")
                
                cert_count = len([g for p in timeline.values() for g in p if g['type'] == 'certification'])
                if cert_count > 0:
                    f.write(f"Working toward {cert_count} certification goal(s)\n")
            
            # Store summary for report
            all_plans_summary[engineer] = {
                'critical_priorities': critical_priorities,
                'growth_priorities': growth_priorities[:2],  # Top 2
                'certification_priorities': [p for p in cert_priorities if 'expiry_date' in p],  # Expired ones
                'immediate_goals': len(timeline['immediate']),
                'short_term_goals': len(timeline['short_term']),
                'long_term_goals': len(timeline['long_term']),
                'total_actions': len(actions)
            }
        
        # Generate a summary report of all IDPs
        with open('development_plans/IDP_Summary_Report.txt', 'w') as f:
            f.write("INDIVIDUAL DEVELOPMENT PLANS - SUMMARY REPORT\n")
            f.write("============================================\n\n")
            
            f.write("Overview:\n")
            f.write(f"- Total IDPs Generated: {len(all_plans_summary)}\n")
            
            # Count critical gaps being addressed
            critical_addressed = sum(len(data['critical_priorities']) for data in all_plans_summary.values())
            f.write(f"- Critical Skill Gaps Being Addressed: {critical_addressed}\n")
            
            # Count certification priorities
            cert_priorities = sum(len(data['certification_priorities']) for data in all_plans_summary.values())
            f.write(f"- Certification Renewals Needed: {cert_priorities}\n\n")
            
            f.write("Individual Summaries:\n")
            for engineer, data in all_plans_summary.items():
                f.write(f"\n{engineer}:\n")
                f.write(f"- Critical Priorities: {len(data['critical_priorities'])}\n")
                f.write(f"- Growth Opportunities: {len(data['growth_priorities'])}\n")
                f.write(f"- Certification Needs: {len(data['certification_priorities'])}\n")
                f.write(f"- Total Development Actions: {data['total_actions']}\n")
                
                if data['critical_priorities']:
                    f.write("  Key Focus Areas: ")
                    f.write(", ".join([p['technology'] for p in data['critical_priorities']]))
                    f.write("\n")
        
        print(f"Generated {len(self.engineers)} individual development plans in the 'development_plans' directory")
        return all_plans_summary


    def run_full_analysis(self):
    
        # Generate visualizations
        self.technology_coverage_heatmap()
        skill_distribution = self.skill_distribution_analysis()
        
        # Generate certification visualizations if available
        if self.cert_df is not None:
            self.certification_coverage_heatmap()
            cert_analysis = self.certification_analysis()
            skill_cert_correlation = self.skill_cert_correlation()
        else:
            cert_analysis = None
            skill_cert_correlation = None
        
        # Generate skill gap analysis
        gap_analysis = self.generate_skill_gap_analysis()
        
        # Generate training recommendations
        training_recommendations = self.generate_training_recommendations()
        
        # Generate individual development plans
        individual_plans = self.generate_individual_development_plans()
        
        # Generate detailed report
        report = self.generate_detailed_report()
        
        return {
            'skill_distribution': skill_distribution,
            'certification_analysis': cert_analysis,
            'skill_cert_correlation': skill_cert_correlation,
            'gap_analysis': gap_analysis,
            'training_recommendations': training_recommendations,
            'individual_plans': individual_plans,
            'report': report
        }
    
    
# Example usage
if __name__ == "__main__":
    # CSV file paths
    skills_csv_path = 'team_skills_data.csv'
    certifications_csv_path = 'team_certifications_data.csv'  # Optional
    
    # Create dashboard instance
    dashboard = EnhancedTechnicalDashboard(skills_csv_path, certifications_csv_path)
    
    # Run full analysis
    results = dashboard.run_full_analysis()
    
    print("\nAnalysis complete. Check the generated files:")
    print("- technology_coverage_heatmap.png")
    print("- certification_coverage_heatmap.png (if cert data provided)")
    print("- technology_avg_skills.png")
    print("- certification_distribution.png (if cert data provided)")
    print("- skill_gap_analysis.png")
    print("- individual_skill_cert_summary.csv")
    print("- team_skill_cert_stats.txt")
    print("- training_recommendations.txt")

# Sample certification data CSV structure
"""
Engineer,Certification,Status,Date
User 1,Cisco Security Certification,Obtained,2023-05-15
User 1,Cloud Security Professional,In Progress,2024-01-20
User 2,Cisco Security Certification,Expired,2022-11-30
User 3,Cisco Security Certification,Obtained,2023-09-10
User 3,Security+ Certification,Obtained,2023-02-28
User 4,Cloud Security Professional,In Progress,2024-02-15
"""
